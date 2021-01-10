"""data.py - data loading"""


import json
import os

import pytorch_lightning as pl
import torch

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
        with open(penne.ASSETS_DIR / name / 'partition.json') as file:
            self.stems = json.load(file)[partition]

    def __getitem__(self, index):
        """Retrieve the indexth item"""
        stem = self.stems[index]

        # TODO - Load from stem
        raise NotImplementedError

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
        # TODO - second argument must be the name of your train partition
        return loader(self.name, 'train', self.batch_size, self.num_workers)

    def val_dataloader(self):
        """Retrieve the PyTorch DataLoader for validation"""
        # TODO - second argument must be the name of your valid partition
        return loader(self.name, 'valid', self.batch_size, self.num_workers)

    def test_dataloader(self):
        """Retrieve the PyTorch DataLoader for testing"""
        # TODO - second argument must be the name of your test partition
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
    raise NotImplementedError
