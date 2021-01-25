"""data.py - data loading"""


import json
import os

import pytorch_lightning as pl
import torch
import numpy as np
import random

import penne


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
        # Get list of stems
        self.stems = partitions(name)[partition]
        self.name = name

    def __getitem__(self, index):
        """Retrieve the indexth item"""
        stem = self.stems[index]

        filepath = stem_to_file(self.name, stem)
        audio, sample_rate = penne.load.audio(filepath)

        # resample
        hop_length = sample_rate // 100
        if sample_rate != penne.SAMPLE_RATE:
            audio = penne.resample(audio, sample_rate)
            hop_length = int(hop_length * penne.SAMPLE_RATE / sample_rate)

        truth = stem_to_truth(self.name, stem)
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
        dataset=Dataset(dataset, partition),
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
    num_frames = 100
    hop_size = penne.SAMPLE_RATE // 100
    features, targets = zip(*batch)
    col_features = []
    col_targets = []
    for i in range(len(targets)):
        audio = features[i]
        target = targets[i]
        frames = torch.nn.functional.unfold(
                audio[:, None, None, :],
                kernel_size=(1, penne.WINDOW_SIZE),
                stride=(1, hop_size))
        curr_frames = min(frames.shape[2], target.shape[1])
        if curr_frames >= num_frames:
            start = random.randint(0, curr_frames - num_frames)
            frames = frames[:,:,start:start+num_frames]
            target = target[:,start:start+num_frames]
        else:
            # pad zeros
            pass
        col_features.append(frames)
        col_targets.append(target)
        
    return (torch.cat(col_features), torch.cat(col_targets))


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


def stem_to_file(name, stem):
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
        return PTDB_stem_to_file(directory, stem)

    raise ValueError(f'Dataset {name} is not implemented')

def MDB_stem_to_file(directory, stem):
    return directory / 'audio_stems' / (stem + ".RESYN.wav")

def PTDB_stem_to_file(directory, stem):
    sub_folder = stem[:3]
    gender = 'FEMALE' if sub_folder[0] == "F" else 'MALE'
    return directory / gender / 'MIC' / sub_folder / ("mic_" + stem + ".wav")

def stem_to_truth(name, stem):
    """Resolve stem to a file in the dataset

    Arguments
        name - string
            The name of the dataset
        stem - string
            The stem representing one item in the dataset

    Returns
        truth - numpy array
            The ground truth frequencies in a numpy array
    """
    directory = penne.DATA_DIR / name

    if name == 'MDB':
        return MDB_stem_to_truth(directory, stem)
    elif name == 'PTDB':
        return PTDB_stem_to_truth(directory, stem)

    raise ValueError(f'Dataset {name} is not implemented')

def MDB_stem_to_truth(directory, stem):
    truth_path = directory / 'annotation_stems' / (stem + ".RESYN.csv")
    arr = np.loadtxt(open(truth_path), delimiter=',')[:,1]
    hopsize = 128 / 44100
    curr = 0.01
    mask = []
    for i in range(0, len(arr)):
        if i * hopsize >= curr:
            curr += 0.01
            mask.append(i)
    return torch.tensor(np.copy(arr[mask]))[None]


def PTDB_stem_to_truth(directory, stem):
    sub_folder = stem[:3]
    gender = 'FEMALE' if sub_folder[0] == "F" else 'MALE'
    truth_path = directory / gender / 'REF' / sub_folder / ("ref_" + stem + ".f0")
    arr = np.loadtxt(open(truth_path), delimiter=' ')[:,0]
    # 32 ms window size, 10 ms hop size
    return torch.tensor(np.copy(arr))[None]