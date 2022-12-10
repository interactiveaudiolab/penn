import math

import torch

import penn


###############################################################################
# Methods for extracting a periodicity estimate from pitch posteriorgram logits
###############################################################################


def entropy(logits):
    """Entropy-based periodicity"""
    distribution = torch.nn.functional.softmax(logits, dim=1)
    return (
        1 + 1 / math.log(penn.PITCH_BINS) * \
        (distribution * torch.log(distribution + 1e-7)).sum(dim=1))


def max(logits):
    """Periodicity as the maximum confidence"""
    if penn.LOSS == 'binary_cross_entropy':
        return torch.sigmoid(logits).max(dim=1).values
    elif penn.LOSS == 'categorical_cross_entropy':
        return torch.nn.functional.softmax(
            logits, dim=1).max(dim=1).values
    raise ValueError(f'Loss function {penn.LOSS} is not implemented')


def sum(logits):
    """Periodicity as the sum of the distribution

    This is really just for PYIN, which performs a masking of the distribution
    probabilities so that it does not always add to one.
    """
    return torch.clip(torch.exp(logits).sum(dim=1), 0, 1)
