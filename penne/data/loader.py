import torch

import penne


def loaders(dataset):
    """Retrieve data loaders for training and evaluation"""
    return loader(dataset, 'train'), loader(dataset, 'valid')


def loader(dataset, partition):
    """Retrieve a data loader"""
    return torch.utils.data.DataLoader(
        dataset=penne.data.Dataset(dataset, partition),
        batch_size=penne.BATCH_SIZE,
        shuffle=partition == 'train',
        num_workers=penne.NUM_WORKERS,
        pin_memory=True)
