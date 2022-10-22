"""dataset.py - data loading"""


from bisect import bisect
import functools

import numpy as np
import torch

import penne


###############################################################################
# Dataset
###############################################################################


class Dataset(torch.utils.data.Dataset):
    """PyTorch dataset

    Arguments
        names - list[str]
            The names of datasets to load from
        partition - string
            The name of the data partition
    """

    def __init__(self, names, partition):
        self.partition = partition
        self.datasets = [Metadata(name, partition) for name in names]

    def __getitem__(self, index):
        """Retrieve the indexth item"""
        # Get dataset to query
        i = 0
        dataset = self.datasets[i]
        upper_bound = len(dataset.frames - (penne.NUM_TRAINING_FRAMES - 1))
        while index > upper_bound:
            i += 1
            dataset[i] = self.names[i]
            upper_bound += len(self.stems[dataset])

        # Get index into dataset
        index -= (upper_bound - len(self.stems[dataset]))

        # Get stem
        stem_index = bisect.bisect_left(dataset.offsets, index)
        stem = self.stems[dataset][stem_index]

        # Get start and end frames
        start = index - dataset.offsets[stem_index]
        end = start + penne.NUM_TRAINING_FRAMES

        # Load from cache
        directory = penne.CACHE_DIR / dataset.name
        audio = np.load(directory / f'{stem}-audio.npy', mmap_mode='r')
        pitch = np.load(directory / f'{stem}-pitch.npy', mmap_mode='r')
        voiced = np.load(directory / f'{stem}-voiced.npy', mmap_mode='r')

        # Make sure lengths match up
        assert penne.convert.samples_to_frames(len(audio)) == len(pitch) == len(voiced)

        # Slice and convert to torch
        audio = torch.from_numpy(
            audio[start * penne.HOPSIZE:end * penne.HOPSIZE])[None]
        pitch = torch.from_numpy(pitch[start:end])[None]
        voiced = torch.from_numpy(voiced[start:end])[None]

        # Convert to pitch bin categories
        bins = penne.convert.frequency_to_bins(pitch)

        return audio, pitch, bins, voiced

    @functools.cached_property
    def __len__(self):
        """Length of the dataset"""
        # Get total number of files
        files = sum(len(dataset.files) for dataset in self.datasets)

        # Get total number of frames
        frames = sum(sum(dataset.frames) for dataset in self.datasets)

        # Get number of valid start points for analysis window
        return frames - (len(files) * (penne.NUM_TRAINING_FRAMES - 1))


###############################################################################
# Metadata
###############################################################################


class Metadata:

    def __init__(self, name, partition):
        self.name = name
        self.stems = penne.data.partitions(name)[partition]
        self.files = [f'{stem}-audio.npy' for stem in self.stems]

        # Get number of frames in each file
        self.frames = [
            penne.convert.samples_to_frames(len(np.load(file, mmap_mode='r')))
            for file in self.files]

        # We require all files to be at least as large as the analysis window
        assert all(frame >= penne.NUM_TRAINING_FRAMES for frame in self.frames)

        # File frame offsets
        self.offsets = np.cumsum(self.frames)
