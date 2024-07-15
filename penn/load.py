import json
import warnings

import torchaudio

import penn


def audio(file):
    """Load audio from disk"""
    audio, sample_rate = torchaudio.load(file)

    # If audio is stereo, convert to mono
    if audio.size(0) == 2:
        warnings.warn(f'Converting stereo audio to mono: {file}')
        audio = audio.mean(dim=0, keepdim=True)

    # Maybe resample
    return penn.resample(audio, sample_rate)


def partition(dataset):
    """Load partitions for dataset"""
    with open(penn.PARTITION_DIR / f'{dataset}.json') as file:
        return json.load(file)
