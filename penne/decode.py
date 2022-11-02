import torch

import penne


###############################################################################
# Decode pitch contour from logits of pitch posteriorgram
###############################################################################


def argmax(logits):
    """Decode pitch using argmax"""
    # Get pitch bins
    bins = logits.argmax(dim=1)

    # Convert to hz
    pitch = penne.convert.bins_to_frequency(bins)

    return bins, pitch


def weighted(logits, window=5):
    """Decode pitch using a normal assumption around the argmax"""
    logits = logits[0].T

    # Get center bins
    bins = logits.argmax(dim=1)

    # Get start and end of windows
    start = torch.maximum(
        bins - window // 2,
        torch.tensor(0, device=bins.device))
    end = torch.minimum(
        bins + window // 2 + 1,
        torch.tensor(penne.PITCH_BINS, device=bins.device))

    # Get local distributions
    for i, (s, e) in enumerate(zip(start, end)):
        logits[i, :s] = -float('inf')
        logits[i, e:] = -float('inf')
    distributions = torch.nn.functional.softmax(logits, dim=1)

    # Cache cents map
    if not hasattr(weighted, 'cents') or weighted.cents.device != bins.device:
        weighted.cents = (
            penne.convert.bins_to_cents(0) +
            penne.CENTS_PER_BIN * torch.arange(
                penne.PITCH_BINS,
                device=bins.device)[None])

    # Pitch is expected value in cents
    pitch = penne.convert.cents_to_frequency(
        (distributions * weighted.cents).sum(dim=1, keepdims=True))

    return bins[None], pitch.T
