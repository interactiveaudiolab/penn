import librosa
import numpy as np
import torch
import torchaudio

import penne


def audio(file):
    """Load audio from disk"""
    audio, sample_rate = torchaudio.load(file)

    # Maybe resample
    return penne.resample(audio, sample_rate)


def model(device, checkpoint, pdc=False):
    """Preloads model from disk"""
    # Bind model and capacity
    penne.infer.checkpoint = checkpoint

    if checkpoint.suffix == '.pth':
        penne.infer.model = penne.PDCModel() if pdc else penne.Model()

        # Load weights
        penne.infer.model.load_state_dict(
            torch.load(checkpoint, map_location=device))

    elif checkpoint.suffix =='.ckpt':
        penne.infer.model = penne.PDCModel() if pdc else penne.Model()

        # Load weights
        penne.infer.model.load_state_dict(
            torch.load(checkpoint, map_location=device)['model_state_dict'])

    else:
        raise ValueError(f'Invalid checkpoint extension for {checkpoint}')

    # Place on device
    penne.infer.model = penne.infer.model.to(torch.device(device))

    # Eval mode
    penne.infer.model.eval()
