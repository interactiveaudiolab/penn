import numpy as np
import torch
import torchaudio
import tqdm

import penne


###############################################################################
# Crepe pitch prediction
###############################################################################


def from_audio(
    audio,
    sample_rate,
    hop_length=None,
    fmin=None,
    fmax=None,
    model=None,
    checkpoint=None,
    batch_size=None,
    device='cpu'):
    """Performs pitch estimation

    Arguments
        audio (torch.tensor [shape=(1, time)])
            The audio signal
        sample_rate (int)
            The sampling rate in Hz
        hop_length (int)
            The hop_length in samples
        fmin (float)
            The minimum allowable frequency in Hz
        fmax (float)
            The maximum allowable frequency in Hz
        model (string)
            The name of the model
        checkpoint (Path)
            The checkpoint file
        batch_size (int)
            The number of frames per batch
        device (string)
            The device used to run inference

    Returns
        pitch (torch.tensor [shape=(1, 1 + int(time // hop_length))])
        periodicity (torch.tensor [shape=(1, 1 + int(time // hop_length))])
    """
    # TODO
    pass


###############################################################################
# Utilities
###############################################################################


def iterator(iterable, message, length=None):
    """Create a tqdm iterator"""
    length = len(iterable) if length is None else length
    return tqdm.tqdm(
        iterable,
        desc=message,
        dynamic_ncols=True,
        total=length)


def resample(audio, sample_rate, target_rate=penne.SAMPLE_RATE):
    """Perform audio resampling"""
    if sample_rate == target_rate:
        return audio
    resampler = torchaudio.transforms.Resample(sample_rate, target_rate)
    resampler = resampler.to(audio.device)
    return resampler(audio)


def entropy(distribution):
    return 1 - (1 / np.log2(penne.PITCH_BINS)) * ((distribution * torch.log2(distribution)).sum())
    #Assumes one frame
    #TODO: sum only over dimension for one frame
