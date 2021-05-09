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

    def __init__(self, name, partition):
        self.name = name
        
        # read information from cache directory
        subfolder = 'voiceonly' if penne.VOICE_ONLY else 'all'
        with open(penne.CACHE_DIR / subfolder / name / "offsets.json", 'r') as f:
            offset_json = json.load(f)
            self.stems = list(offset_json[partition].keys())
            self.offsets = []
            for stem in self.stems:
                self.offsets.append(offset_json[partition][stem][0])
            # sort by offset
            self.offsets, self.stems = zip(*(sorted(zip(self.offsets, self.stems))))
            self.total_nframes = offset_json['totals'][partition]

    def __getitem__(self, index):
        try:
            """Retrieve the indexth item"""
            stem_idx = bisect.bisect_right(self.offsets, index) - 1
            stem = self.stems[stem_idx]
            frame_idx = index - self.offsets[stem_idx]
            frames = np.load(penne.data.stem_to_cache_frames(self.name, stem, penne.VOICE_ONLY), mmap_mode='r')
            frame = frames[:,:,frame_idx]
            if frame.dtype == np.int16:
                frame = frame.astype(np.float32) / np.iinfo(np.int16).max
            frame = torch.from_numpy(frame.copy())

            if penne.WHITEN:
                frame -= frame.mean(dim=1, keepdim=True)
                frame /= torch.max(torch.tensor(1e-10, device=frame.device),
                    frame.std(dim=1, keepdim=True))

            annotation_path = stem_to_cache_annotation(self.name, stem, penne.VOICE_ONLY)
            annotations = penne.load.annotation_from_cache(annotation_path)
            annotation = annotations[:,frame_idx]

            if annotation == 0:
                annotation[0] = torch.randint(0, penne.PITCH_BINS, annotation.shape)

            return (frame, annotation)
        except Exception as e:
            print(e)
            print(index)

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
        return loader(self.name, 'train', self.batch_size, self.num_workers)

    def val_dataloader(self):
        """Retrieve the PyTorch DataLoader for validation"""
        return loader(self.name, 'valid', self.batch_size, self.num_workers)

    def test_dataloader(self):
        """Retrieve the PyTorch DataLoader for testing"""
        return loader(self.name, 'test', self.batch_size, self.num_workers)


###############################################################################
# Data loader
###############################################################################

def loader(dataset, partition, batch_size=64, num_workers=None):
    """Retrieve a data loader"""
    return torch.utils.data.DataLoader(
        dataset=Dataset(dataset, partition, partition != 'test' or penne.CHUNK_BATCH),
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
    return (col_features, col_targets)

def old_collate_fn(batch):
    """Turns __getitem__ output into a batch ready for inference

    Arguments
        batch - list
            The outputs of __getitem__ for each item in batch

    Returns
        collated - tuple
            The input features and ground truth targets ready for inference
    """
    # TODO - Perform any necessary padding or slicing to ensure that input
    #        features and output targets can be concatenated. Then,
    #        concatenate them and return them as torch tensors. See
    #        https://pytorch.org/docs/stable/data.html#dataloader-collate-fn
    #        for more information on the collate function (note that
    #        automatic batching is enabled).
    num_frames = penne.convert.seconds_to_frames(1)
    features, targets = zip(*batch)
    col_features = []
    col_targets = []
    for i in range(len(targets)):
        audio = features[i]
        target = targets[i]
        frames = torch.nn.functional.unfold(
                audio[:, None, None, :],
                kernel_size=(1, penne.WINDOW_SIZE),
                stride=(1, penne.HOP_SIZE))
        curr_frames = min(frames.shape[2], target.shape[1])
        if curr_frames > num_frames:
            if penne.CHUNK_BATCH:
                start = random.randint(0, curr_frames - num_frames)
                frames = frames[:,:,start:start+num_frames]
                target = target[:,start:start+num_frames]
            else:
                random_frames = random.sample(range(curr_frames), num_frames)
                frames = frames[:,:,random_frames]
                target = target[:,random_frames]
        else:
            # no files are shorter than 1 second (100 frames), so we don't need to pad
            pass
        col_features.append(frames)
        col_targets.append(target)
    
    col_features = torch.cat(col_features)
    col_targets = torch.cat(col_targets)
    col_features = col_features.permute(0, 2, 1).reshape((len(batch)*num_frames, penne.WINDOW_SIZE))

    if penne.WHITEN:
        # Mean-center
        col_features -= col_features.mean(dim=1, keepdim=True)
        # Scale
        # Note: during silent frames, this produces very large values. But
        # this seems to be what the network expects.
        col_features /= torch.max(torch.tensor(1e-10, device=col_features.device),
                    col_features.std(dim=1, keepdim=True))

    col_targets = col_targets.reshape((len(batch)*num_frames,))
    unvoiced = col_targets==0
    col_targets[col_targets==0] = torch.randint(0, penne.PITCH_BINS, col_targets[col_targets==0].shape)
    return (col_features, col_targets, unvoiced)


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
    """Resolve stem to a truth numpy array in the dataset

    Arguments
        name - string
            The name of the dataset
        path - Path
            The path to the file

    Returns
        stem - Path
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
    """Resolve stem to a file in the dataset

    Arguments
        name - string
            The name of the dataset
        stem - string
            The stem representing one item in the dataset

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