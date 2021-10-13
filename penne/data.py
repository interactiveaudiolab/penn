"""data.py - data loading"""

import json
import os

import pytorch_lightning as pl
import torch
import numpy as np
import random
import bisect

import penne
from scipy.io import wavfile

###############################################################################
# Dataset
###############################################################################

class Dataset(torch.utils.data.Dataset):
    """PyTorch dataset

    Arguments
        name - string
            The name of the dataset
        partition - string
            The name of the data partition
    """

    def __init__(self, name, partition, voiceonly=penne.VOICE_ONLY):
        self.name = name
        self.voiceonly = voiceonly
        self.stems = {}
        self.offsets = {}
        
        # read information from cache directory
        subfolder = 'voiceonly' if voiceonly else 'all'
        self.split = 0
        self.total_nframes = 0
        if self.name == 'BOTH':
            for dataset_name in ['MDB', 'PTDB']:
                with open(penne.CACHE_DIR / subfolder / dataset_name / "offsets.json", 'r') as f:
                    offset_json = json.load(f)
                    self.stems[dataset_name] = list(offset_json[partition].keys())
                    self.offsets[dataset_name] = [offset_json[partition][stem][0] for stem in self.stems[dataset_name]]
                    # sort by offset
                    self.offsets[dataset_name], self.stems[dataset_name] = zip(*(sorted(zip(self.offsets[dataset_name], self.stems[dataset_name]))))
                    self.total_nframes += offset_json['totals'][partition]
                if self.split == 0:
                    self.split = self.total_nframes
        elif name in ['MDB', 'PTDB']:
            with open(penne.CACHE_DIR / subfolder / self.name / "offsets.json", 'r') as f:
                offset_json = json.load(f)
                self.stems[self.name] = list(offset_json[partition].keys())
                self.offsets[self.name] = [offset_json[partition][stem][0] for stem in self.stems[self.name]]
                # sort by offset
                self.offsets[self.name], self.stems[self.name] = zip(*(sorted(zip(self.offsets[self.name], self.stems[self.name]))))
                self.total_nframes = offset_json['totals'][partition]
        else:
            raise ValueError("Dataset name must be MDB, PTDB, or BOTH")

    def __getitem__(self, index):
        if self.name in ['MDB', 'PTDB']:
            return self.getitem_from_dataset(self.name, index)
        else:
            if index < self.split:
                return self.getitem_from_dataset('MDB', index)
            else:
                return self.getitem_from_dataset('PTDB', index - self.split)


    def getitem_from_dataset(self, name, index):
        """Retrieve the indexth item"""
        # get the stem that indexth item is from
        stem_idx = bisect.bisect_right(self.offsets[name], index) - 1
        stem = self.stems[name][stem_idx]

        # get samples in indexth frame
        frame_idx = index - self.offsets[name][stem_idx]
        frames = np.load(penne.data.stem_to_cache_frames(name, stem, self.voiceonly), mmap_mode='r')
        frame = frames[:,:,frame_idx]
        # Convert to float32
        if frame.dtype == np.int16:
            frame = frame.astype(np.float32) / np.iinfo(np.int16).max
        frame = torch.from_numpy(frame.copy())

        # optionally normalize
        if penne.WHITEN or penne.ORIGINAL_CREPE:
            frame -= frame.mean(dim=1, keepdim=True)
            frame /= torch.max(torch.tensor(1e-10, device=frame.device),
                frame.std(dim=1, keepdim=True))

        # get the annotation bin
        annotation_path = stem_to_cache_annotation(name, stem, self.voiceonly)
        annotations = penne.load.annotation_from_cache(annotation_path)
        annotation = annotations[:,frame_idx]

        # choose a random bin if unvoiced
        if annotation == 0:
            annotation[0] = torch.randint(0, penne.PITCH_BINS, annotation.shape)
        # insert shape comment
        return (frame, annotation)
        

    def __len__(self):
        """Length of the dataset"""
        return self.total_nframes

###############################################################################
# Data module
###############################################################################

class DataModule(pl.LightningDataModule):
    """PyTorch Lightning data module

    Arguments
        name - string
            The name of the dataset
        batch_size - int
            The size of a batch
        num_workers - int or None
            Number data loading jobs to launch. If None, uses num cpu cores.
    """

    def __init__(self, name, batch_size=64, num_workers=None):
        super().__init__()
        self.name = name
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        """Retrieve the PyTorch DataLoader for training"""
        return loader(self.name, 'train', self.batch_size, self.num_workers, penne.VOICE_ONLY)

    def val_dataloader(self):
        """Retrieve the PyTorch DataLoader for validation"""
        return loader(self.name, 'valid', self.batch_size, self.num_workers, True)

    def test_dataloader(self):
        """Retrieve the PyTorch DataLoader for testing"""
        return loader(self.name, 'test', self.batch_size, self.num_workers, True)

###############################################################################
# Data loader
###############################################################################

def loader(dataset, partition, batch_size=64, num_workers=None, voiceonly=penne.VOICE_ONLY):
    """Retrieve a data loader"""
    if dataset == 'BOTH':
        dataset_obj = torch.utils.data.ConcatDataset([
            Dataset('MDB', partition, voiceonly),
            Dataset('PTDB', partition, voiceonly)
        ])
    else:
        dataset_obj = Dataset(dataset, partition, voiceonly)
    return torch.utils.data.DataLoader(
        dataset=dataset_obj,
        batch_size=batch_size,
        shuffle='test' not in partition,
        num_workers=os.cpu_count() if num_workers is None else num_workers,
        pin_memory=True,
        collate_fn=collate_fn)

###############################################################################
# Collate function
###############################################################################

def collate_fn(batch):
    features, targets = zip(*batch)
    col_features = torch.cat(list(features))
    col_targets = torch.cat(list(targets))
    # insert comment about shapes of these
    return (col_features, col_targets)

###############################################################################
# Utilities
###############################################################################

def partitions(name):
    """Retrieve the data partitions for a dataset

    Arguments
        name - string
            The dataset name

    Returns
        partitions - dict(string, list(string))
            The dataset partitions. The key is the partition name and the
            value is the list of stems belonging to that partition.
    """
    if not hasattr(partitions, name):
        with open(penne.ASSETS_DIR / name / 'partition.json') as file:
            setattr(partitions, name, json.load(file))
    return getattr(partitions, name)

def stem_to_file(name, stem, filetype='audio'):
    """Resolve stem to a file in the dataset

    Arguments
        name - string
            The name of the dataset
        stem - string
            The stem representing one item in the dataset
        filetype - string
            Either 'audio' or 'laryn' for the mic or laryngograph PTDB file respectively

    Returns
        file - Path
            The corresponding file
    """
    directory = penne.DATA_DIR / name

    if name == 'MDB':
        return MDB_stem_to_file(directory, stem)
    elif name == 'PTDB':
        return PTDB_stem_to_file(directory, stem, filetype)

    raise ValueError(f'Dataset {name} is not implemented')

def MDB_stem_to_file(directory, stem):
    return directory / 'audio_stems' / (stem + ".RESYN.wav")

def PTDB_stem_to_file(directory, stem, filetype='audio'):
    sub_folder = stem[:3]
    gender = 'FEMALE' if sub_folder[0] == "F" else 'MALE'
    if filetype == 'audio':
        return directory / gender / 'MIC' / sub_folder / ("mic_" + stem + ".wav")
    if filetype == 'laryn':
        return directory / gender / 'LAR' / sub_folder / ("lar_" + stem + ".wav")
    raise ValueError("Filetype doesn't exist")

def stem_to_annotation(name, stem):
    """Resolve stem to a truth numpy array in the dataset

    Arguments
        name - string
            The name of the dataset
        stem - string
            The stem representing one item in the dataset

    Returns
        file - Path
            The corresponding file
    """
    directory = penne.DATA_DIR / name

    if name == 'MDB':
        return MDB_stem_to_annotation(directory, stem)
    elif name == 'PTDB':
        return PTDB_stem_to_annotation(directory, stem)

    raise ValueError(f'Dataset {name} is not implemented')

def MDB_stem_to_annotation(directory, stem):
    return directory / 'annotation_stems' / (stem + ".RESYN.csv")

def PTDB_stem_to_annotation(directory, stem):
    # This file contains a four column matrix which includes the pitch, a voicing decision, the 
    # root mean square values and the peak-normalized autocorrelation values respectively
    # (https://www2.spsc.tugraz.at//databases/PTDB-TUG/DOCUMENTATION/PTDB-TUG_REPORT.pdf)
    sub_folder = stem[:3]
    gender = 'FEMALE' if sub_folder[0] == "F" else 'MALE'
    return directory / gender / 'REF' / sub_folder / ("ref_" + stem + ".f0")

def file_to_stem(name, path):
    """Resolve file path to the stem associated with that file

    Arguments
        name - string
            The name of the dataset
        path - Path
            The path to the file

    Returns
        stem - string
            The stem of the file
    """
    if name == 'MDB':
        return MDB_file_to_stem(path)
    elif name == 'PTDB':
        return PTDB_file_to_stem(path)

    raise ValueError(f'Dataset {name} is not implemented')

def MDB_file_to_stem(path):
    return path.stem.split('.')[0]

def PTDB_file_to_stem(path):
    return path.stem[path.stem.index('_')+1:]

def stem_to_cache_file(name, stem, filetype='audio', voiceonly=False):
    """Resolve stem to a file in the cache

    Arguments
        name - string
            The name of the dataset
        stem - string
            The stem representing one item in the dataset
        voiceonly - bool
            If True, get only voiced frames, else get all frames

    Returns
        file - Path
            The corresponding file
    """
    subfolder = 'voiceonly' if voiceonly else 'all'
    directory = penne.CACHE_DIR / subfolder / name

    if name == 'MDB':
        return MDB_stem_to_cache_file(directory, stem)
    elif name == 'PTDB':
        return PTDB_stem_to_cache_file(directory, stem, filetype)

    raise ValueError(f'Dataset {name} is not implemented')

def MDB_stem_to_cache_file(directory, stem):
    return directory / 'audio' / (stem + ".RESYN.wav")

def PTDB_stem_to_cache_file(directory, stem, filetype='audio'):
    if filetype == 'audio':
        return directory / 'audio' / ("mic_" + stem + ".wav")
    if filetype == 'laryn':
        return directory / 'audio' / ("lar_" + stem + ".wav")
    raise ValueError("Filetype doesn't exist")

def stem_to_cache_annotation(name, stem, voiceonly=False):
    """Resolve stem to a truth numpy array in the cache

    Arguments
        name - string
            The name of the dataset
        stem - string
            The stem representing one item in the dataset
        voiceonly - bool
            If True, get only voiced frames, else get all frames

    Returns
        file - Path
            The corresponding file
    """
    subfolder = 'voiceonly' if voiceonly else 'all'
    directory = penne.CACHE_DIR / subfolder / name

    if name == 'MDB':
        return MDB_stem_to_cache_annotation(directory, stem)
    elif name == 'PTDB':
        return PTDB_stem_to_cache_annotation(directory, stem)

    raise ValueError(f'Dataset {name} is not implemented')

def MDB_stem_to_cache_annotation(directory, stem):
    return directory / 'annotation' / (stem + ".RESYN.npy")

def PTDB_stem_to_cache_annotation(directory, stem):
    # This file contains a four column matrix which includes the pitch, a voicing decision, the 
    # root mean square values and the peak-normalized autocorrelation values respectively
    # (https://www2.spsc.tugraz.at//databases/PTDB-TUG/DOCUMENTATION/PTDB-TUG_REPORT.pdf)
    return directory / 'annotation' / ("ref_" + stem + ".npy")

def stem_to_cache_frames(name, stem, voiceonly=False):
    """Resolve stem to a numpy array of frames in the cache

    Arguments
        name - string
            The name of the dataset
        stem - string
            The stem representing one item in the dataset
        voiceonly - bool
            If True, get only voiced frames, else get all frames

    Returns
        file - Path
            The corresponding file
    """
    subfolder = 'voiceonly' if voiceonly else 'all'
    directory = penne.CACHE_DIR / subfolder / name
    if name == 'MDB':
        return MDB_stem_to_cache_frames(directory, stem)
    elif name == 'PTDB':
        return PTDB_stem_to_cache_frames(directory, stem)

    raise ValueError(f'Dataset {name} is not implemented')

def MDB_stem_to_cache_frames(directory, stem):
    return directory / 'frames' / (stem + ".RESYN.npy")

def PTDB_stem_to_cache_frames(directory, stem):
    return directory / 'frames' / ("mic_" + stem + ".npy")