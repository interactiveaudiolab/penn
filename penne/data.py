"""data.py - data loading"""


import json
import os

import pytorch_lightning as pl
import torch
import numpy as np
import random

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

    def __init__(self, name, partition, random_slice=False):
        # Get list of stems
        self.stems = partitions(name)[partition]
        self.name = name
        self.random_slice = random_slice

    def __getitem__(self, index):
        """Retrieve the indexth item"""
        stem = self.stems[index]

        filepath = stem_to_file(self.name, stem)
        sample_rate, audio = wavfile.read(filepath, mmap=True)

        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / np.iinfo(np.int16).max

        hop_length = sample_rate // 100

        # TEMPORARY
        # REMOVE THIS
        # FOR OVERFIT BATCHES
        # random.seed(0)

        if self.random_slice:
            resampled_nsamples = int(len(audio) * penne.SAMPLE_RATE / sample_rate)
            if self.name == 'MDB':
                nframes = penne.convert.samples_to_frames(resampled_nsamples)
                start_index = random.randint(0, nframes-penne.convert.seconds_to_frames(1)-1)
                start = hop_length * start_index
                length = sample_rate
                audio = audio[start:start+length]
            elif self.name == 'PTDB':
                resampled_window_size = int(penne.WINDOW_SIZE * sample_rate / penne.SAMPLE_RATE)
                nframes = penne.convert.samples_to_frames(resampled_nsamples-penne.WINDOW_SIZE)
                start_index = random.randint(0, nframes-penne.convert.seconds_to_frames(1)-penne.convert.samples_to_frames(penne.WINDOW_SIZE)-1)
                start = resampled_window_size // 2 + hop_length * start_index
                length = sample_rate + resampled_window_size
                audio = audio[start:start+length]
            else:
                raise ValueError(f'Dataset {name} is not implemented')
            

        # resample
        if sample_rate != penne.SAMPLE_RATE:
            # torch audio resampling?
            audio = penne.resample(torch.tensor(np.copy(audio))[None], sample_rate)
            hop_length = int(hop_length * penne.SAMPLE_RATE / sample_rate)

        if self.name == 'MDB':
            audio = torch.nn.functional.pad(audio, (penne.WINDOW_SIZE//2, penne.WINDOW_SIZE//2))

        annotation_path = stem_to_annotation(self.name, stem)

        truth = penne.load.pitch_annotation(self.name, annotation_path)
        if self.random_slice:
            resampled_start = int(start/sample_rate * penne.SAMPLE_RATE)
            if self.name == 'MDB':
                start = penne.convert.samples_to_frames(resampled_start)
            elif self.name == 'PTDB':
                start = penne.convert.samples_to_frames(resampled_start-penne.WINDOW_SIZE // 2)
            truth = truth[:,start:start+int(penne.SAMPLE_RATE/penne.HOP_SIZE)+1].long()
        
        return (audio, truth)

    def __len__(self):
        """Length of the dataset"""
        return len(self.stems)


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
        dataset=Dataset(dataset, partition, partition != 'test'),
        batch_size=batch_size,
        shuffle='train' in partition,
        num_workers=os.cpu_count() if num_workers is None else num_workers,
        pin_memory=True,
        collate_fn=collate_fn)


###############################################################################
# Collate function
###############################################################################


def collate_fn(batch):
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
            start = random.randint(0, curr_frames - num_frames)
            frames = frames[:,:,start:start+num_frames]
            target = target[:,start:start+num_frames]
        else:
            # no files are shorter than 1 second (100 frames), so we don't need to pad
            pass
        col_features.append(frames)
        col_targets.append(target)
    
    col_features = torch.cat(col_features)
    col_targets = torch.cat(col_targets)
    col_features = col_features.permute(0, 2, 1).reshape((len(batch)*num_frames, penne.WINDOW_SIZE))
    col_targets = col_targets.reshape((len(batch)*num_frames,))
    col_targets[col_targets==0] = torch.randint(0, penne.PITCH_BINS, col_targets[col_targets==0].shape)
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
    # annotation = np.loadtxt(open(truth_path), delimiter=',')
    # xp, fp = annotation[:,0], annotation[:,1]
    # # original annotations are spaced every 128 / 44100 seconds; we downsample to 0.01 seconds
    # hopsize = 128 / 44100
    # interpx = np.arange(0, hopsize*len(xp), 0.01)
    # new_annotation = np.interp(interpx, xp, fp)
    # return torch.tensor(np.copy(new_annotation))[None]


def PTDB_stem_to_annotation(directory, stem):
    # This file contains a four column matrix which includes the pitch, a voicing decision, the 
    # root mean square values and the peak-normalized autocorrelation values respectively
    # (https://www2.spsc.tugraz.at//databases/PTDB-TUG/DOCUMENTATION/PTDB-TUG_REPORT.pdf)
    sub_folder = stem[:3]
    gender = 'FEMALE' if sub_folder[0] == "F" else 'MALE'
    return directory / gender / 'REF' / sub_folder / ("ref_" + stem + ".f0")
    # arr = np.loadtxt(open(truth_path), delimiter=' ')[:,0]
    # # 32 ms window size, 10 ms hop size
    # return torch.tensor(np.copy(arr))[None]

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