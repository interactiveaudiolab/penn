import os

import numpy as np
import torch
import penne
from scipy.io import wavfile
import librosa


def audio(filename):
    """Load audio from disk"""
    sample_rate, audio = wavfile.read(filename)

    # Convert to float32
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / np.iinfo(np.int16).max

    # PyTorch is not compatible with non-writeable arrays, so we make a copy
    return torch.tensor(np.copy(audio))[None], sample_rate


def model(device, checkpoint=penne.FULL_CHECKPOINT, pdc=False):
    """Preloads model from disk"""
    # Bind model and capacity
    penne.infer.checkpoint = checkpoint

    if checkpoint.suffix == '.pth':
        penne.infer.model = penne.PDCModel() if pdc else penne.Model()

        # Load weights
        penne.infer.model.load_state_dict(
            torch.load(checkpoint, map_location=device))

    elif checkpoint.suffix =='.ckpt':
        if pdc:
            penne.infer.model = penne.PDCModel.load_from_checkpoint(checkpoint)
        else:
            penne.infer.model = penne.Model.load_from_checkpoint(checkpoint)
    else:
        raise ValueError(f'Invalid checkpoint extension for {checkpoint}')
    
    # Place on device
    penne.infer.model = penne.infer.model.to(torch.device(device))
    
    # Eval mode
    penne.infer.model.eval()

def annotation_from_cache(path):
    return torch.from_numpy(np.load(path))

def pitch_annotation(name, path, bins=True):
    if name == 'MDB':
        return MDB_pitch(path, bins)
    elif name == 'PTDB':
        return PTDB_pitch(path, bins)
    else:
        ValueError(f'Dataset {name} is not implemented')

def MDB_pitch(path, bins=True):
    annotation = np.loadtxt(open(path), delimiter=',')
    xp, fp = annotation[:,0], annotation[:,1]
    # original annotations are spaced every 128 / 44100 seconds; we downsample to 0.01 seconds
    hopsize = 128 / 44100
    
    # get duration of original file
    duration = librosa.get_duration(filename=penne.data.stem_to_file('MDB', penne.data.file_to_stem('MDB', path)))
    
    sequence, voiced = linear_interp(fp)
    # linearly interpolate at 0.01 second intervals
    interpx = 0.01 * np.arange(0, penne.convert.seconds_to_frames(duration))
    new_annotation = 2 ** np.interp(interpx, xp, np.log2(sequence))
    new_voiced = np.interp(interpx, xp, voiced)
    new_voiced = new_voiced > 0.5

    tensor_annotation = torch.tensor(np.copy(new_annotation))[None]
    tensor_voiced = torch.tensor(np.copy(new_voiced))[None]

    if not bins:
        return tensor_annotation, tensor_voiced
    
    # convert frequency annotations to bins
    bin_annotation = penne.convert.frequency_to_bins(tensor_annotation)
    bin_annotation[~tensor_voiced] = 0
    return bin_annotation

def PTDB_pitch(path, bins=True):
    # PTDB annotations are extracted using RAPT with 32 ms window size, 10 ms hop size
    arr = np.loadtxt(open(path), delimiter=' ')[:,0]

    if not bins:
        sequence, voiced = linear_interp(arr)
        tensor_annotation = torch.tensor(np.copy(sequence))[None]
        tensor_voiced = torch.tensor(np.copy(voiced))[None]
        return tensor_annotation, tensor_voiced
    
    # convert frequency annotations to bins
    bin_annotation = penne.convert.frequency_to_bins(torch.tensor(np.copy(arr))[None])
    bin_annotation[bin_annotation < 0] = 0
    return bin_annotation

def linear_interp(sequence):
    unvoiced = sequence == 0
    sequence = np.log2(sequence)
    sequence[unvoiced] = np.interp(
        np.where(unvoiced)[0], np.where(~unvoiced)[0], sequence[~unvoiced])
    return 2**sequence, ~unvoiced