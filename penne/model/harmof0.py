import math
import torch
import torchaudio

import penne


###############################################################################
# Harmf0
###############################################################################


class Harmof0(torch.nn.Sequential):

    def __init__(self, channels=[32, 64, 128, 128]):
        super().__init__(
            LogHarmonicSpectrogram(),
            Block(1, channels[0], dilation_mode='harmonic'),
            Block(channels[0], channels[1]),
            Block(channels[1], channels[2]),
            Block(channels[2], channels[3]),
            torch.nn.Conv2d(channels[3], channels[3] // 2, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channels[3] // 2, 1, kernel_size=1))

    def forward(self, audio):
        # shape=(batch, 1, penne.NUM_TRAINING_SAMPLES) =>
        # shape=(batch, penne.PITCH_BINS, penne.NUM_TRAINING_FRAMES)
        return super().forward(audio).squeeze(1)


###############################################################################
# Utilities
###############################################################################


class Block(torch.nn.Sequential):

    def __init__(self, in_channels, out_channels, dilation_mode='fixed'):
        super().__init__(
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(3, 3),
                padding=(1, 1)),
            torch.nn.ReLU(),
            HarmonicDilatedConvolution(out_channels) \
                if dilation_mode == 'harmonic' else torch.nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=(1, 3),
                    padding='same',
                    dilation=[1, penne.OCTAVE // penne.CENTS_PER_BIN]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels))


class HarmonicDilatedConvolution(torch.nn.Sequential):

    def __init__(self, channels):
        dilations = \
            torch.log(torch.arange(1, 13)) / \
            math.log(2 ** (1.0 / (penne.OCTAVE / penne.CENTS_PER_BIN)))
        super().__init__(
            MultiRateDilatedConvolution(
                channels,
                channels,
                dilations=dilations.round().to(torch.int)))


class MultiRateDilatedConvolution(torch.nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        dilations=[0, 12, 19, 24, 28, 31, 34, 36]):
        super().__init__()
        self.dilations = dilations
        self.convs = torch.nn.ModuleList([
            torch.nn.Conv2d(in_channels, out_channels, kernel_size)
            for _ in range(len(dilations))])

    def forward(self, spectrogram):
        activation = self.convs[0](spectrogram)
        activation = torch.nn.functional.pad(
            activation,
            (0, self.dilations[0]))
        activation = activation[:, :, :, self.dilations[0]:]
        for conv, dilation in zip(self.convs[1:], self.dilations[1:]):
            x = conv(spectrogram)[:, :, :, dilation:]
            activation[:, :, :, :x.shape[3]] += x
        return activation


class LogHarmonicSpectrogram(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # Create window
        window = torch.hann_window(penne.WINDOW_SIZE)[None, None, :]
        self.register_buffer("window", window, persistent=False)

        # Create harmonic bin spacing
        resolution = penne.SAMPLE_RATE / penne.NUM_FFT
        idxs = torch.arange(0, penne.PITCH_BINS)
        bins_per_octave = penne.OCTAVE / penne.CENTS_PER_BIN
        log_idxs = penne.FMIN * (2 ** (idxs / bins_per_octave)) / resolution
        log_idxs_floor = torch.floor(log_idxs).long()
        log_idxs_floor_w = (
            log_idxs - log_idxs_floor).reshape([1, 1, penne.PITCH_BINS])
        log_idxs_ceiling = torch.ceil(log_idxs).long()
        log_idxs_ceiling_w = (
            log_idxs_ceiling - log_idxs).reshape([1, 1, penne.PITCH_BINS])
        self.register_buffer(
            "log_idxs_floor",
            log_idxs_floor,
            persistent=False)
        self.register_buffer(
            "log_idxs_floor_w",
            log_idxs_floor_w,
            persistent=False)
        self.register_buffer(
            "log_idxs_ceiling",
            log_idxs_ceiling,
            persistent=False)
        self.register_buffer(
            "log_idxs_ceiling_w",
            log_idxs_ceiling_w,
            persistent=False)

        # Create amplitude scaling
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)

    def forward(self, audio):
        # Chunk audio
        frames = torch.nn.functional.unfold(
            audio[:, None],
            kernel_size=(1, penne.WINDOW_SIZE),
            stride=(1, penne.HOPSIZE)).permute(0, 2, 1)

        # Compute short-time fourier transform
        stft =  torch.fft.fft(frames * self.window)

        # Take the real components
        real = torch.abs(stft[:, :, :penne.NUM_FFT // 2 + 1])

        # Compute magnitude spectrogram
        magnitude = real * real

        # Apply harmonic bin spacing
        harmonic = \
            magnitude[:, :, self.log_idxs_floor] * self.log_idxs_floor_w + \
            magnitude[:, :, self.log_idxs_ceiling] * self.log_idxs_ceiling_w

        # Apply log-based amplitude scaling
        return self.amplitude_to_db(harmonic.permute(0, 2, 1))[:, None]
