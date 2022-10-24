import json

import torchaudio

import penne


def audio(file):
    """Load audio from disk"""
    audio, sample_rate = torchaudio.load(file)

    # Maybe resample
    return penne.resample(audio, sample_rate)


def partition(dataset):
    """Load partitions for dataset"""
    with open(penne.PARTITION_DIR / f'{dataset}.json') as file:
        return json.load(file)
