"""dataset.py - data loading"""


import bisect
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
        self.datasets = [Metadata(name, partition) for name in names]

    def __getitem__(self, index):
        """Retrieve the indexth item"""
        # Get dataset to query
        i = 0
        dataset = self.datasets[i]
        upper_bound = dataset.total
        while index > upper_bound:
            i += 1
            dataset = self.datasets[i]
            upper_bound += dataset.total

        # Get index into dataset
        index -= (upper_bound - dataset.total)

        # Get stem
        stem_index = bisect.bisect(dataset.offsets, index)
        stem = dataset.stems[stem_index]

        # Get start and end frames
        start = \
            index - (0 if stem_index == 0 else dataset.offsets[stem_index - 1])
        end = start + penne.NUM_TRAINING_FRAMES

        # Get start and end samples
        start_sample = start * penne.HOPSIZE
        end_sample = start_sample + penne.NUM_TRAINING_SAMPLES

        # Load from cache
        directory = penne.CACHE_DIR / dataset.name
        audio = np.load(directory / f'{stem}-audio.npy', mmap_mode='r')
        pitch = np.load(directory / f'{stem}-pitch.npy', mmap_mode='r')
        voiced = np.load(directory / f'{stem}-voiced.npy', mmap_mode='r')

        # Slice and convert to torch
        audio = torch.from_numpy(
            audio[start_sample:end_sample].copy())[None]
        pitch = torch.from_numpy(pitch[start:end].copy())[None]
        voiced = torch.from_numpy(voiced[start:end].copy())[None]

        # Convert to pitch bin categories
        bins = penne.convert.frequency_to_bins(pitch)

        # Set unvoiced bins to random values
        bins = torch.where(
            ~voiced,
            torch.randint(0, penne.PITCH_BINS, bins.shape, dtype=torch.long),
            bins)

        return audio, bins, pitch, voiced, stem

    def __len__(self):
        """Length of the dataset"""
        return sum(dataset.total for dataset in self.datasets)


###############################################################################
# Metadata
###############################################################################


class Metadata:

    def __init__(self, name, partition):
        self.name = name
        self.stems = penne.load.partition(name)[partition]
        self.files = [
            penne.CACHE_DIR / name / f'{stem}-audio.npy'
            for stem in self.stems]

        # Get number of frames in each file
        self.frames = [
            penne.convert.samples_to_frames(len(np.load(file, mmap_mode='r')))
            for file in self.files]

        # We require all files to be at least as large as the analysis window
        num_frames = max(
            penne.NUM_TRAINING_FRAMES,
            1 + penne.NUM_TRAINING_SAMPLES // penne.HOPSIZE)
        assert all(frame >= num_frames for frame in self.frames)

        # Remove invalid start points
        self.frames = [frame - (num_frames - 1) for frame in self.frames]

        # Save frame offsets
        self.offsets = np.cumsum(self.frames)

        # Total number of valid start points
        self.total = self.offsets[-1]
