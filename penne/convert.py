import scipy
import torch

import penne


###############################################################################
# Pitch conversions
###############################################################################


def bins_to_cents(bins):
    """Converts pitch bins to cents"""
    cents = penne.CENTS_PER_BIN * bins

    # Maybe trade quantization error for noise
    return dither(cents) if penne.DITHER else cents


def bins_to_frequency(bins):
    """Converts pitch bins to frequency in Hz"""
    return cents_to_frequency(bins_to_cents(bins))


def cents_to_bins(cents, quantize_fn=torch.floor):
    """Converts cents to pitch bins"""
    bins = quantize_fn(cents / penne.CENTS_PER_BIN).int()
    bins[bins < 0] = 0
    bins[bins >= penne.PITCH_BINS] = penne.PITCH_BINS - 1
    return bins


def cents_to_frequency(cents):
    """Converts cents to frequency in Hz"""
    return penne.FMIN * 2 ** (cents / penne.OCTAVE)


def frequency_to_bins(frequency, quantize_fn=torch.floor):
    """Convert frequency in Hz to pitch bins"""
    return cents_to_bins(frequency_to_cents(frequency), quantize_fn)


def frequency_to_cents(frequency):
    """Convert frequency in Hz to cents"""
    return penne.OCTAVE * torch.log2(frequency / penne.FMIN)


def frequency_to_samples(frequency, sample_rate=penne.SAMPLE_RATE):
    """Convert frequency in Hz to number of samples per period"""
    return int(sample_rate / frequency)


###############################################################################
# Time conversions
###############################################################################


def frames_to_samples(frames):
    """Convert number of frames to samples"""
    return frames * penne.HOPSIZE


def frames_to_seconds(frames):
    """Convert number of frames to seconds"""
    return frames * penne.HOPSIZE_SECONDS


def seconds_to_frames(seconds):
    """Convert seconds to number of frames"""
    return samples_to_frames(seconds_to_samples(seconds))


def seconds_to_samples(seconds):
    """Convert seconds to number of samples"""
    return int(seconds * penne.SAMPLE_RATE)


def samples_to_frames(samples):
    """Convert samples to number of frames"""
    return samples // penne.HOPSIZE


def samples_to_seconds(samples):
    """Convert number of samples to seconds"""
    return samples / penne.SAMPLE_RATE


###############################################################################
# Utilities
###############################################################################


def dither(cents):
    """Dither the predicted pitch in cents to remove quantization error"""
    noise = scipy.stats.triang.rvs(
        c=0.5,
        loc=-penne.CENTS_PER_BIN,
        scale=2 * penne.CENTS_PER_BIN,
        size=cents.size())
    return cents + cents.new_tensor(noise)
