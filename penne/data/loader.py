import torch

import penne


def loader(datasets, partition, gpu=None):
    """Retrieve a data loader"""
    # Create dataset
    dataset = penne.data.Dataset(datasets, partition)

    # Create sampler
    sampler = penne.data.sampler(dataset, partition)

    # Get batch size
    if partition == 'train':

        # Maybe split batch over GPUs
        if torch.distributed.is_initialized():
            batch_size = penne.BATCH_SIZE // torch.distributed.get_world_size()
        else:
            batch_size = penne.BATCH_SIZE

    elif partition == 'valid':
        batch_size = penne.BATCH_SIZE
    elif partition == 'test':
        batch_size = 1
    else:
        raise ValueError(f'Partition {partition} is not defined')

    # Create data loader
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=penne.NUM_WORKERS,
        pin_memory=gpu is not None,
        sampler=sampler)
