import numpy as np
import torbi
import torch

import penn


###############################################################################
# Decode pitch contour from logits of pitch posteriorgram
###############################################################################


def argmax(logits):
    """Decode pitch using argmax"""
    # Get pitch bins
    bins = logits.argmax(dim=1)

    # Convert to hz
    pitch = penn.convert.bins_to_frequency(bins)

    return bins, pitch


def viterbi(logits):
    """Decode pitch using viterbi decoding (from librosa)"""
    # Normalize and convert to numpy
    if penn.METHOD == 'pyin':
        periodicity = penn.periodicity.sum(logits).T
        unvoiced = (
            (1 - periodicity) / penn.PITCH_BINS).repeat(penn.PITCH_BINS, 1)
        distributions = torch.cat(
            (torch.exp(logits.permute(2, 1, 0)), unvoiced[None]),
            dim=1)

    else:

        # Viterbi REQUIRES a categorical distribution, even if the loss was BCE
        distributions = torch.nn.functional.softmax(logits, dim=1)
        distributions = distributions.permute(2, 1, 0)
        distributions = distributions.to(device='cpu', dtype=torch.float32)

    # Cache viterbi probabilities
    if not hasattr(viterbi, 'transition'):
        xx, yy = torch.meshgrid(
            torch.arange(penn.PITCH_BINS),
            torch.arange(penn.PITCH_BINS),
            indexing='ij')
        bins_per_octave = penn.OCTAVE / penn.CENTS_PER_BIN
        max_octaves_per_frame = \
            penn.MAX_OCTAVES_PER_SECOND * penn.HOPSIZE / penn.SAMPLE_RATE
        max_bins_per_frame = max_octaves_per_frame * bins_per_octave + 1
        transition = torch.clip(max_bins_per_frame - (xx - yy).abs(), 0)
        transition = transition / transition.sum(dim=1, keepdims=True)
        viterbi.transition = transition

        if penn.METHOD == 'pyin':

            # Add unvoiced probabilities
            viterbi.transition = torch.kron(
                torch.tensor([[.99, .01], [.01, .99]]),
                viterbi.transition)

            # Uniform initial probabilities
            viterbi.initial = torch.zeros(2 * penn.PITCH_BINS)
            viterbi.initial[penn.PITCH_BINS:] = 1 / penn.PITCH_BINS

        else:

            # Uniform initial probabilities
            viterbi.initial = torch.full(
                (penn.PITCH_BINS,),
                1 / penn.PITCH_BINS)

    # Viterbi decoding
    bins = torbi.decode(
        distributions[0].T,
        viterbi.transition,
        viterbi.initial,
        penn.VITERBI_MAX_CHUNK_SIZE
    )[:, None]

    # Convert to frequency in Hz
    if penn.DECODER.endswith('normal'):

        # Decode using an assumption of normality around to the viterbi path
        bins = bins.to(logits.device).T
        pitch = local_expected_value_from_bins(bins, logits).T

    elif penn.METHOD == 'pyin':

        # Linearly interpolate unvoiced regions
        pitch = penn.convert.bins_to_frequency(bins)
        pitch[bins >= penn.PITCH_BINS] = 0
        pitch = penn.data.preprocess.interpolate_unvoiced(pitch.numpy())[0]
        pitch = torch.from_numpy(pitch).to(logits.device)
        bins = bins.to(logits.device)

    else:

        # Argmax decoding
        bins = bins.to(logits.device).T
        pitch = penn.convert.bins_to_frequency(bins)

    return bins.T, pitch.T


def local_expected_value(logits, window=penn.LOCAL_PITCH_WINDOW_SIZE):
    """Decode pitch using a normal assumption around the argmax"""
    # Get center bins
    bins = logits.argmax(dim=1)

    return bins, local_expected_value_from_bins(bins, logits, window)


###############################################################################
# Utilities
###############################################################################


def expected_value(logits, cents):
    """Expected value computation from logits"""
    # Get local distributions
    if penn.LOSS == 'categorical_cross_entropy':
        distributions = torch.nn.functional.softmax(logits, dim=1)
    elif penn.LOSS == 'binary_cross_entropy':
        distributions = torch.sigmoid(logits)
    else:
        raise ValueError(f'Loss {penn.LOSS} is not defined')

    # Pitch is expected value in cents
    pitch = (distributions * cents).sum(dim=1, keepdims=True)

    # BCE requires normalization
    if penn.LOSS == 'binary_cross_entropy':
        pitch = pitch / distributions.sum(dim=1)

    # Convert to hz
    return penn.convert.cents_to_frequency(pitch)


def local_expected_value_from_bins(
    bins,
    logits,
    window=penn.LOCAL_PITCH_WINDOW_SIZE):
    """Decode pitch using normal assumption around argmax from bin indices"""
    # Pad
    padded = torch.nn.functional.pad(
        logits.squeeze(2),
        (window // 2, window // 2),
        value=-float('inf'))

    # Get indices
    indices = \
        bins.repeat(1, window) + torch.arange(window, device=bins.device)[None]

    # Get values in cents
    cents = penn.convert.bins_to_cents(torch.clip(indices - window // 2, 0))

    # Decode using local expected value
    return expected_value(torch.gather(padded, 1, indices), cents)
