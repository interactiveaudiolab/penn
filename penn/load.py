import json

import torchaudio

import penn


def audio(file):
    """Load audio from disk"""
    audio, sample_rate = torchaudio.load(file)

    # Maybe resample
    return penn.resample(audio, sample_rate)


def partition(dataset):
    """Load partitions for dataset"""
    with open(penn.PARTITION_DIR / f'{dataset}.json') as file:
        return json.load(file)
