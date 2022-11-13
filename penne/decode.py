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
    pitch = penne.convert.bins_to_frequency(bins)

    return bins, pitch


def viterbi(logits):
    """Decode pitch using viterbi decoding (from librosa)"""
    # Normalize and convert to numpy
    if penne.METHOD == 'pyin':
        periodicity = penne.periodicity.sum(logits)
        unvoiced = (
            (1 - periodicity) / penne.PITCH_BINS).repeat(penne.PITCH_BINS, 1)
        distributions = torch.cat(
            (torch.exp(logits), unvoiced[None]),
            dim=1).numpy()
    else:
        distributions = torch.nn.functional.softmax(logits, dim=1)[0].numpy()

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
    pitch = penne.convert.bins_to_frequency(bins)

    if penne.METHOD == 'pyin':

        # Linearly interpolate unvoiced regions
        pitch[bins >= penne.PITCH_BINS] = 0
        pitch = penne.data.preprocess.interpolate_unvoiced(pitch.numpy())[0]
        pitch = torch.from_numpy(pitch)

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
