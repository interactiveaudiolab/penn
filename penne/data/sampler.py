import math

import torch

import penne


###############################################################################
# Batch sampler
###############################################################################


def sampler(dataset, partition):
    """Create batch index sampler"""
    # Get sampler indices
    indices = (
        dataset.voiced_indices() if penne.VOICED_ONLY and partition == 'train'
        else list(range(len(dataset))))

    # Maybe use distributed sampler for training
    if partition == 'train':
        return (
            DistributedSampler(indices) if torch.distributed.is_initialized()
            else Sampler(indices))

    # Always use (deterministic) random sampler for validation
    elif partition == 'valid':
        return Sampler(indices)

    # Sample test data sequentially
    elif partition == 'test':
        return torch.utils.data.SequentialSampler(dataset)


###############################################################################
# Custom samplers
###############################################################################


class Sampler:

    def __init__(self, indices):
        self.indices = indices
        self.epoch = 0

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(penne.RANDOM_SEED + self.epoch)
        for i in torch.randperm(len(self.indices), generator=generator):
            yield self.indices[i]

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        self.epoch = epoch


class DistributedSampler:

    def __init__(self, indices):
        self.indices = indices
        self.rank = torch.distributed.get_rank()
        self.num_replicas = torch.distributed.get_world_size()
        self.num_samples = math.ceil(len(self.indices) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = self.indices.copy()

        # Deterministic shuffling based on epoch
        generator = torch.Generator()
        generator.manual_seed(penne.RANDOM_SEED + self.epoch)
        indices = torch.randperm(indices, generator=self.generator).tolist()

        # Add extra samples to make it evenly divisible
        padding = self.total_size - len(indices)
        if padding <= len(indices):
            indices += indices[:padding]
        else:
            indices += (
                indices * math.ceil(padding / len(indices)))[:padding]
        assert len(indices) == self.total_size

        # Subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch