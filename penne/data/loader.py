import torch

import penne


def loader(datasets, partition, gpu=None):
    """Retrieve a data loader"""
    return torch.utils.data.DataLoader(
        dataset=penne.data.Dataset(datasets, partition),
        batch_size=penne.BATCH_SIZE,
        shuffle=partition == 'train',
        num_workers=penne.NUM_WORKERS,
        pin_memory=gpu is not None)
