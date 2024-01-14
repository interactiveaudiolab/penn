import abc
import functools

import numpy as np
import torbi
import torch

import penn


###############################################################################
# Base pitch posteriorgram decoder
###############################################################################


class Decoder(abc.ABC):
    """Base decoder"""

    def __init__(self, local_expected_value=True):
        self.local_expected_value = local_expected_value

    @abc.abstractmethod
    def __call__(self, logits):
        """Perform decoding"""
        pass


###############################################################################
# Derived pitch posteriorgram decoders
###############################################################################


class Argmax(Decoder):
    """Decode pitch using argmax"""

    def __init__(self, local_expected_value=penn.LOCAL_EXPECTED_VALUE):
        super().__init__(local_expected_value)

    def __call__(self, logits):
        # Get pitch bins
        bins = logits.argmax(dim=1)

        # Convert to frequency in Hz
        if self.local_expected_value:

            # Decode using an assumption of normality around the argmax path
            pitch = local_expected_value_from_bins(bins, logits)

        else:

            # Linearly interpolate unvoiced regions
            pitch = penn.convert.bins_to_frequency(bins)

        return bins, pitch


class PYIN(Decoder):
    """Decode pitch via peak picking + Viterbi. Used by PYIN."""

    def __init__(self, local_expected_value=False):
        super().__init__(local_expected_value)

    def __call__(self, logits):
        """PYIN decoding"""
        periodicity = penn.periodicity.sum(logits).T
        unvoiced = (
            (1 - periodicity) / penn.PITCH_BINS).repeat(penn.PITCH_BINS, 1)
        distributions = torch.cat(
            (torch.exp(logits.permute(2, 1, 0)), unvoiced[None]),
            dim=1)

        # Viterbi decoding
        gpu = (
            None if distributions.device.type == 'cpu'
            else distributions.device.index)
        bins = torbi.decode(
            distributions[0].T,
            self.transition(),
            self.initial(),
            gpu)

        # Convert to frequency in Hz
        if self.local_expected_value:

            # Decode using an assumption of normality around the viterbi path
            pitch = local_expected_value_from_bins(bins.T, logits).T

        else:

            # Argmax decoding
            pitch = penn.convert.bins_to_frequency(bins)

        # Linearly interpolate unvoiced regions
        pitch[bins >= penn.PITCH_BINS] = 0
        pitch = penn.data.preprocess.interpolate_unvoiced(pitch.numpy())[0]
        pitch = torch.from_numpy(pitch).to(logits.device)
        bins = bins.to(logits.device)

        return bins.T, pitch.T

    @functools.cached_property
    def initial(self):
        """Create initial probability matrix for PYIN"""
        initial = torch.zeros(2 * penn.PITCH_BINS)
        initial[penn.PITCH_BINS:] = 1 / penn.PITCH_BINS

    @functools.cached_property
    def transition(self):
        """Create the Viterbi transition matrix for PYIN"""
        transition = triangular_transition_matrix()

        # Add unvoiced probabilities
        transition = torch.kron(
            torch.tensor([[.99, .01], [.01, .99]]),
            transition)


class Viterbi(Decoder):

    def __init__(self, local_expected_value=True):
        super().__init__(local_expected_value)

    def __call__(self, logits):
        """Decode pitch using viterbi decoding (from librosa)"""
        distributions = torch.nn.functional.softmax(logits, dim=1)
        distributions = distributions.permute(2, 1, 0)

        # Viterbi decoding
        gpu = (
            None if distributions.device.type == 'cpu'
            else distributions.device.index)
        bins = torbi.decode(
            distributions[0].T,
            self.transition(),
            self.initial(),
            gpu)

        # Convert to frequency in Hz
        if self.local_expected_value:

            # Decode using an assumption of normality around the viterbi path
            pitch = local_expected_value_from_bins(bins.T, logits).T

        else:

            # Argmax decoding
            pitch = penn.convert.bins_to_frequency(bins)

        return bins.T, pitch.T

    @functools.cached_property
    def transition(self):
        """Create uniform initial probabilities"""
        return torch.full((penn.PITCH_BINS,), 1 / penn.PITCH_BINS)

    @functools.cached_property
    def transition(self):
        """Create Viterbi transition probability matrix"""
        return triangular_transition_matrix()


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


def triangular_transition_matrix():
    """Create a triangular distribution transition matrix"""
    xx, yy = torch.meshgrid(
        torch.arange(penn.PITCH_BINS),
        torch.arange(penn.PITCH_BINS),
        indexing='ij')
    bins_per_octave = penn.OCTAVE / penn.CENTS_PER_BIN
    max_octaves_per_frame = \
        penn.MAX_OCTAVES_PER_SECOND * penn.HOPSIZE / penn.SAMPLE_RATE
    max_bins_per_frame = max_octaves_per_frame * bins_per_octave + 1
    transition = torch.clip(max_bins_per_frame - (xx - yy).abs(), 0)
    return transition / transition.sum(dim=1, keepdims=True)
