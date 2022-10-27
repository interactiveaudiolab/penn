import torch

import penne


def entropy(logits):
    """Entropy-based periodicity"""
    distribution = torch.nn.functional.softmax(logits, dim=1)
    return (
        1 - (1 / torch.log2(penne.PITCH_BINS)) * \
        ((distribution * torch.log2(distribution)).sum(dim=1)))


def max(logits):
    """Periodicity as the maximum confidence"""
    if penne.LOSS == 'binary_cross_entropy':
        return torch.sigmoid(logits).max(dim=1).values
    elif penne.LOSS == 'categorical_cross_entropy':
        return torch.nn.functional.softmax(
            logits, dim=1).max(dim=1).values
    else:
        raise ValueError(f'Loss function {penne.LOSS} is not implemented')
