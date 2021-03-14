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


def model(device, checkpoint=penne.FULL_CHECKPOINT):
    """Preloads model from disk"""
    # Bind model and capacity
    penne.infer.checkpoint = checkpoint

    if checkpoint.suffix == '.pth':
        penne.infer.model = penne.Model()

        # Load weights
        penne.infer.model.load_state_dict(
            torch.load(checkpoint, map_location=device))

    elif checkpoint.suffix =='.ckpt':
        penne.infer.model = penne.Model.load_from_checkpoint(checkpoint)
    else:
        raise ValueError(f'Invalid checkpoint extension for {checkpoint}')
    
    # Place on device
    penne.infer.model = penne.infer.model.to(torch.device(device))
    
    # Eval mode
    penne.infer.model.eval()

def pitch_annotation(name, path):
    if name == 'MDB':
        return MDB_pitch(path)
    elif name == 'PTDB':
        return PTDB_pitch(path)
    else:
        ValueError(f'Dataset {name} is not implemented')

def MDB_pitch(path):
    annotation = np.loadtxt(open(path), delimiter=',')
    xp, fp = annotation[:,0], annotation[:,1]
    # original annotations are spaced every 128 / 44100 seconds; we downsample to 0.01 seconds
    hopsize = 128 / 44100
    duration = librosa.get_duration(filename=penne.data.stem_to_file('MDB', penne.data.file_to_stem('MDB', path)))
    interpx = 0.01 * np.arange(0, penne.convert.seconds_to_frames(duration))
    new_annotation = np.interp(interpx, xp, fp)
    bin_annotation = penne.convert.frequency_to_bins(torch.tensor(np.copy(new_annotation))[None])
    bin_annotation[bin_annotation < 0] = 0
    return bin_annotation

def PTDB_pitch(path):
    arr = np.loadtxt(open(path), delimiter=' ')[:,0]
    # 32 ms window size, 10 ms hop size
    bin_annotation = penne.convert.frequency_to_bins(torch.tensor(np.copy(arr))[None])
    bin_annotation[bin_annotation < 0] = 0
    return bin_annotation