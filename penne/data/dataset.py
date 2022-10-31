import bisect

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
        if self.partition == 'test':
            return self.load_inference(index)
        return self.load_training(index)

    def __len__(self):
        """Length of the dataset"""
        if self.partition == 'test':
            return sum(len(dataset.files) for dataset in self.datasets)
        return sum(dataset.total for dataset in self.datasets)

    def load_inference(self, index):
        """Load item for inference"""
        # Get dataset to query
        i = 0
        dataset = self.datasets[i]
        upper_bound = len(dataset.files)
        while index >= upper_bound:
            i += 1
            dataset = self.datasets[i]
            upper_bound += len(dataset.files)

        # Get index into dataset
        index -= (upper_bound - len(dataset.files))

        # Get stem
        stem = dataset.stems[index]

        # Load from cache
        directory = penne.CACHE_DIR / dataset.name
        audio = np.load(directory / f'{stem}-audio.npy')
        pitch = np.load(directory / f'{stem}-pitch.npy')
        voiced = np.load(directory / f'{stem}-voiced.npy')

        # Convert to torch
        audio = torch.from_numpy(audio)[None]
        pitch = torch.from_numpy(pitch)
        voiced = torch.from_numpy(voiced)

        # Convert to pitch bin categories
        bins = penne.convert.frequency_to_bins(pitch)

        # Set unvoiced bins to random values
        bins = torch.where(
            ~voiced,
            torch.randint(0, penne.PITCH_BINS, bins.shape, dtype=torch.long),
            bins)

        return audio, bins, pitch, voiced, stem

    def load_training(self, index):
        """Load item for training"""
        # Get dataset to query
        i = 0
        dataset = self.datasets[i]
        upper_bound = dataset.total
        while index >= upper_bound:
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
        start_sample = \
            penne.convert.frames_to_samples(start) - \
            (penne.WINDOW_SIZE - penne.HOPSIZE) // 2
        end_sample = start_sample + penne.NUM_TRAINING_SAMPLES

        # Load from cache
        directory = penne.CACHE_DIR / dataset.name
        waveform = np.load(directory / f'{stem}-audio.npy', mmap_mode='r')
        pitch = np.load(directory / f'{stem}-pitch.npy', mmap_mode='r')
        voiced = np.load(directory / f'{stem}-voiced.npy', mmap_mode='r')

        # Slice audio
        if start_sample < 0:
            audio = torch.zeros(
                (penne.NUM_TRAINING_SAMPLES,),
                dtype=torch.float)
            audio[-start_sample:] = torch.from_numpy(
                waveform[:end_sample].copy())
        elif end_sample > len(waveform):
            audio = torch.zeros(
                (penne.NUM_TRAINING_SAMPLES,),
                dtype=torch.float)
            audio[:len(waveform) - end_sample] = torch.from_numpy(
                waveform[start_sample:].copy())
        else:
            audio = torch.from_numpy(
                waveform[start_sample:end_sample].copy())

        # Slice pitch and voicing
        pitch = torch.from_numpy(pitch[start:end].copy())
        voiced = torch.from_numpy(voiced[start:end].copy())

        # Convert to pitch bin categories
        bins = penne.convert.frequency_to_bins(pitch)

        # Set unvoiced bins to random values
        bins = torch.where(
            ~voiced,
            torch.randint(0, penne.PITCH_BINS, bins.shape, dtype=torch.long),
            bins)

        return audio[None], bins, pitch, voiced, stem

    def voiced_indices(self):
        """Retrieve the indices with voiced start frames"""
        offset = 0
        indices = []
        for dataset in self.datasets:
            indices += [index + offset for index in dataset.voiced_indices()]
            offset += dataset.total
        return indices


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
        assert all(frame >= penne.NUM_TRAINING_FRAMES for frame in self.frames)

        # Remove invalid center points
        self.frames = [
            frame - (penne.NUM_TRAINING_FRAMES - 1) for frame in self.frames]

        # Save frame offsets
        self.offsets = np.cumsum(self.frames)

        # Total number of valid start points
        self.total = self.offsets[-1]

    def voiced_indices(self):
        """Retrieve the indices with voiced start frames"""
        # Get voicing files
        files = [
            penne.CACHE_DIR / self.name / f'{stem}-voiced.npy'
            for stem in self.stems]

        offset = 0
        indices = []
        for file in files:

            # Load
            voiced = np.load(file)

            # Remove invalid center points
            if penne.NUM_TRAINING_FRAMES > 1:
                voiced = voiced[:-(penne.NUM_TRAINING_FRAMES - 1)]

            # Update
            indices.extend(list(voiced.nonzero()[0] + offset))
            offset += len(voiced)

        return indices
