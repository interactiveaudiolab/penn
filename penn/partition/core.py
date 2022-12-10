import json
import random

import penn


###############################################################################
# Dataset-specific
###############################################################################


def datasets(datasets):
    """Partition datasets"""
    for name in datasets:
        dataset(name)


def dataset(name):
    """Partition dataset"""
    # Get dataset stems
    stems = sorted([
        file.stem[:-6] for file in
        (penn.CACHE_DIR / name).glob('*-audio.npy')])
    random.seed(penn.RANDOM_SEED)
    random.shuffle(stems)

    # Get split points
    left, right = int(.70 * len(stems)), int(.85 * len(stems))

    # Perform partition
    partition = {
        'train': sorted(stems[:left]),
        'valid': sorted(stems[left:right]),
        'test': sorted(stems[right:])}

    # Write partition file
    with open(penn.PARTITION_DIR / f'{name}.json', 'w') as file:
        json.dump(partition, file, indent=4)
