import librosa
import numpy as np
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
    # Maybe dither to prevent banding during downstream quantization
    pitch = penne.convert.bins_to_frequency(bins, penne.DITHER)

    return bins, pitch


def viterbi(logits):
    """Decode pitch using viterbi decoding (from librosa)"""
    # Normalize and convert to numpy
    if penne.METHOD == 'pyin':
        periodicity = penne.periodicity.sum(logits).T
        unvoiced = (
            (1 - periodicity) / penne.PITCH_BINS).repeat(penne.PITCH_BINS, 1)
        distributions = torch.cat(
            (torch.exp(logits.permute(2, 1, 0)), unvoiced[None]),
            dim=1).numpy()
    else:

        # Viterbi REQUIRES a categorical distribution, even if the loss was BCE
        distributions = torch.nn.functional.softmax(logits, dim=1)
        distributions = distributions.permute(2, 1, 0).numpy()

    # Cache viterbi probabilities
    if not hasattr(viterbi, 'transition'):
        # Get number of bins per frame
        bins_per_octave = penne.OCTAVE / penne.CENTS_PER_BIN
        max_octaves_per_frame = \
            penne.MAX_OCTAVES_PER_SECOND * penne.HOPSIZE / penne.SAMPLE_RATE
        max_bins_per_frame = max_octaves_per_frame * bins_per_octave+ 1

        # Construct the within voicing transition probabilities
        viterbi.transition = librosa.sequence.transition_local(
            penne.PITCH_BINS,
            max_bins_per_frame,
            window='triangle',
            wrap=False)

        if penne.METHOD == 'pyin':

            # Add unvoiced probabilities
            viterbi.transition = np.kron(
                librosa.sequence.transition_loop(2, .99),
                viterbi.transition)

            # Uniform initial probabilities
            viterbi.initial = np.zeros(2 * penne.PITCH_BINS)
            viterbi.initial[penne.PITCH_BINS:] = 1 / penne.PITCH_BINS

        else:

            # Uniform initial probabilities
            viterbi.initial = np.full(penne.PITCH_BINS, 1 / penne.PITCH_BINS)

    # Viterbi decoding
    bins = librosa.sequence.viterbi(
        distributions,
        viterbi.transition,
        p_init=viterbi.initial)
    bins = torch.from_numpy(bins.astype(np.int32))

    # Convert to frequency in Hz
    if penne.DECODER.endswith('weighted'):

        # Decode using an assumption of normality around to the viterbi path
        pitch = weighted_from_bins(bins, logits)

    else:

        # Maybe dither to prevent banding during downstream quantization
        pitch = penne.convert.bins_to_frequency(bins, penne.DITHER)

    if penne.METHOD == 'pyin':

        # Linearly interpolate unvoiced regions
        pitch[bins >= penne.PITCH_BINS] = 0
        pitch = penne.data.preprocess.interpolate_unvoiced(pitch.numpy())[0]
        pitch = torch.from_numpy(pitch)

    return bins, pitch


def weighted(logits, window=penne.LOCAL_PITCH_WINDOW_SIZE):
    """Decode pitch using a normal assumption around the argmax"""
    # Get center bins
    bins = logits.argmax(dim=1)

    return bins, weighted_from_bins(bins, logits, window)


###############################################################################
# Utilities
###############################################################################


def expected_value(logits, cents):
    """Expected value computation from logits"""
    # Get local distributions
    if penne.LOSS == 'categorical_cross_entropy':
        distributions = torch.nn.functional.softmax(logits, dim=1)
    elif penne.LOSS == 'binary_cross_entropy':
        distributions = torch.sigmoid(logits)
    else:
        raise ValueError(f'Loss {penne.LOSS} is not defined')

    # Pitch is expected value in cents
    pitch = (distributions * cents).sum(dim=1, keepdims=True)

    # BCE requires normalization
    if penne.LOSS == 'binary_cross_entropy':
        pitch = pitch / distributions.sum(dim=1)

    # Convert to hz
    return penne.convert.cents_to_frequency(pitch)


def weighted_from_bins(bins, logits, window=penne.LOCAL_PITCH_WINDOW_SIZE):
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
    cents = penne.convert.bins_to_cents(torch.clip(indices - window // 2, 0))

    # Decode using local expected value
    return expected_value(torch.gather(padded, 1, indices), cents)
