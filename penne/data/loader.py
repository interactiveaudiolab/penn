import torch

import penne


def loader(datasets, partition, gpu=None):
    """Retrieve a data loader"""
    # Create dataset
    dataset = penne.data.Dataset(datasets, partition)

    # Create sampler
    sampler = penne.data.sampler(dataset, partition)

    # Create data loader
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1 if partition == 'test' else penne.BATCH_SIZE,
        num_workers=penne.NUM_WORKERS,
        pin_memory=gpu is not None,
        sampler=sampler)
