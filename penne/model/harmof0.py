import torch
import torchaudio

import penne


###############################################################################
# HarmF0
###############################################################################


class Harmof0(torch.nn.Module):

    def __init__(self,
            n_har=12,
            bins_per_octave=60,
            dilation_modes=['log_scale', 'fixed', 'fixed', 'fixed'],
            dilation_rates=[60, 60, 60, 60],
            channels=[32, 64, 128, 128],
            dil_kernel_sizes= [[1, 3], [1,3], [1,3], [1,3]],
        ):
        super().__init__()
        self.spectrogram = LogHarmonicSpectrogram(bins_per_octave)
        bins = bins_per_octave
        self.block_1 = dila_conv_block(1, channels[0], bins, n_har=n_har, dilation_mode=dilation_modes[0], dilation_rate=dilation_rates[0], dil_kernel_size=dil_kernel_sizes[0], kernel_size=[3, 3], padding=[1,1])
        bins = bins // 2
        self.block_2 = dila_conv_block(channels[0], channels[1], bins, 3, dilation_mode=dilation_modes[1], dilation_rate=dilation_rates[1], dil_kernel_size=dil_kernel_sizes[1], kernel_size=[3, 3], padding=[1,1])
        self.block_3 = dila_conv_block(channels[1], channels[2], bins, 3, dilation_mode=dilation_modes[2], dilation_rate=dilation_rates[2], dil_kernel_size=dil_kernel_sizes[2], kernel_size=[3, 3], padding=[1,1])
        self.block_4 = dila_conv_block(channels[2], channels[3], bins, 3, dilation_mode=dilation_modes[3], dilation_rate=dilation_rates[3], dil_kernel_size=dil_kernel_sizes[3], kernel_size=[3, 3], padding=[1,1])
        self.conv_5 = torch.nn.Conv2d(channels[3], channels[3]//2, kernel_size=[1,1])
        self.conv_6 = torch.nn.Conv2d(channels[3]//2, 1, kernel_size=[1,1])

    def forward(self, waveforms):
        x = self.spectrogram(waveforms.transpose(2,1)).float().unsqueeze(1)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.conv_5(x)
        x = torch.relu(x)
        x = self.conv_6(x)
        x = torch.sigmoid(x)
        return torch.squeeze(x, dim=1)


###############################################################################
# Utilities
###############################################################################


class MRDConv(torch.nn.Module):
    """Multiple Rate Dilated Convolution"""

    def __init__(self, in_channels, out_channels, dilation_list=[0, 12, 19, 24, 28, 31, 34, 36]):
        super().__init__()
        self.dilation_list = dilation_list
        self.conv_list = []
        for i in range(len(dilation_list)):
            self.conv_list += [torch.nn.Conv2d(in_channels, out_channels, kernel_size = [1, 1])]
        self.conv_list = torch.nn.ModuleList(self.conv_list)

    def forward(self, specgram):
        dilation = self.dilation_list[0]
        y = self.conv_list[0](specgram)
        y = torch.nn.functional.pad(y, pad=[0, dilation])
        y = y[:, :, :, dilation:]
        for i in range(1, len(self.conv_list)):
            dilation = self.dilation_list[i]
            x = self.conv_list[i](specgram)
            x = x[:, :, :, dilation:]
            n_freq = x.size()[3]
            y[:, :, :, :n_freq] += x
        return y


class LogHarmonicSpectrogram(torch.nn.Module):

    def __init__(self, bins_per_octave=60, fmin=31.75):
        super().__init__()
        # Create window
        window = torch.hann_window(penne.WINDOW_SIZE)[None, None, :]
        self.register_buffer("window", window, persistent=False)

        # Create harmonic bin spacing
        resolution = penne.SAMPLE_RATE / penne.NUM_FFT
        idxs = torch.arange(0, penne.PITCH_BINS)
        log_idxs = fmin * (2 ** (idxs / bins_per_octave)) / resolution
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

    def forward(self, waveforms):
        specgram =  torch.fft.fft(waveforms * self.window)
        specgram = torch.abs(specgram[:, :, :self.n_fft // 2 + 1])
        specgram = specgram * specgram
        specgram = \
            specgram[:,:, self.log_idxs_floor] * self.log_idxs_floor_w + \
            specgram[:, :, self.log_idxs_ceiling] * self.log_idxs_ceiling_w
        return self.amplitude_to_db(specgram)


############################
# From harmof0: network.py
############################


def dila_conv_block(
    in_channel,
    out_channel,
    bins_per_octave,
    n_har,
    dilation_mode,
    dilation_rate,
    dil_kernel_size,
    kernel_size = [1,3],
    padding = [0,1]):
    conv = torch.nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding)
    batch_norm = torch.nn.BatchNorm2d(out_channel)
    if(dilation_mode == 'log_scale'):
        a = torch.log(torch.arange(1, n_har + 1)) / torch.log(2 ** (1.0 / bins_per_octave))
        dilation_list = a.round().to(torch.int)
        conv_log_dil = MRDConv(out_channel, out_channel, dilation_list)
        return torch.nn.Sequential(
            conv,torch.nn.ReLU(),
            conv_log_dil,torch.nn.ReLU(),
            batch_norm,
        )
    elif(dilation_mode == 'fixed'):
        conv_dil = torch.nn.Conv2d(out_channel, out_channel, kernel_size=dil_kernel_size, padding='same', dilation=[1, dilation_rate])
        return torch.nn.Sequential(
            conv,torch.nn.ReLU(),
            conv_dil,torch.nn.ReLU(),
            batch_norm)
