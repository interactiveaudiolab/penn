import torch

import penne


def entropy(distribution):
    # TODO - make from logits
    return 1 - (1 / torch.log2(penne.PITCH_BINS)) * ((distribution * torch.log2(distribution)).sum())


def max(logits):
    # TODO
    pass
