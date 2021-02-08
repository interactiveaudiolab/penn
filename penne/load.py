import os

import numpy as np
import torch
import penne
from scipy.io import wavfile


def audio(filename):
    """Load audio from disk"""
    sample_rate, audio = wavfile.read(filename)

    # Convert to float32
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / np.iinfo(np.int16).max

    # PyTorch is not compatible with non-writeable arrays, so we make a copy
    return torch.tensor(np.copy(audio))[None], sample_rate


def model(device, capacity='full'):
    """Preloads model from disk"""
    # Bind model and capacity
    penne.infer.capacity = capacity
    penne.infer.model = penne.Model(capacity)

    # Load weights
    file = os.path.join(os.path.dirname(__file__), 'assets', f'{capacity}.pth')
    penne.infer.model.load_state_dict(
        torch.load(file, map_location=device))

    # Place on device
    penne.infer.model = penne.infer.model.to(torch.device(device))

    # Eval mode
    penne.infer.model.eval()

def MDB_pitch(path):
    annotation = np.loadtxt(open(path), delimiter=',')
    xp, fp = annotation[:,0], annotation[:,1]
    # original annotations are spaced every 128 / 44100 seconds; we downsample to 0.01 seconds
    hopsize = 128 / 44100
    interpx = np.arange(0, hopsize*len(xp), 0.01)
    new_annotation = np.interp(interpx, xp, fp)
    return torch.tensor(np.copy(new_annotation))[None]

def PTDB_pitch(path):
    arr = np.loadtxt(open(path), delimiter=' ')[:,0]
    # 32 ms window size, 10 ms hop size
    return torch.tensor(np.copy(arr))[None]