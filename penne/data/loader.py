import torch

import penne


def loader(datasets, partition, gpu=None):
    """Retrieve a data loader"""
    return torch.utils.data.DataLoader(
        dataset=penne.data.Dataset(datasets, partition),
        batch_size=1 if partition == 'test' else penne.BATCH_SIZE,
        shuffle=partition != 'test',
        num_workers=penne.NUM_WORKERS,
        pin_memory=gpu is not None)
