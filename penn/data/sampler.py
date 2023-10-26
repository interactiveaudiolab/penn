import torch

import penn


###############################################################################
# Batch sampler
###############################################################################


def sampler(dataset, partition):
    """Create batch index sampler"""
    # Get sampler indices
    indices = (
        dataset.voiced_indices() if penn.VOICED_ONLY and partition == 'train'
        else list(range(len(dataset))))

    # Maybe use distributed sampler for training
    if partition == 'train':
        return Sampler(indices)

    # Possibly deterministic random sampler for validation
    elif partition == 'valid':
        return Sampler(indices)

    # Sample test data sequentially
    elif partition == 'test':
        return torch.utils.data.SequentialSampler(dataset)

    else:
        raise ValueError(f'Partition {partition} is not implemented')


###############################################################################
# Custom samplers
###############################################################################


class Sampler:

    def __init__(self, indices):
        self.indices = indices
        self.epoch = 0

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(penn.RANDOM_SEED + self.epoch)
        for i in torch.randperm(len(self.indices), generator=generator):
            yield self.indices[i]

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        self.epoch = epoch
