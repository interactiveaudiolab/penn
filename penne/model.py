import functools

import torch
import torch.nn.functional as F
import torchaudio
#import pytorch_lightning as pl

import penne
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import math

###############################################################################
# Model definition
###############################################################################

class Model(torch.nn.Module):
    """PyTorch Lightning model definition"""

    def __init__(self, name='default'):
        super().__init__()

        self.epsilon = 0.0010000000474974513
        self.learning_rate = penne.LEARNING_RATE

        # equivalent to Keras default momentum
        self.momentum = 0.01

        self.name = name
        self.ex_batch = None
        self.best_loss = float('inf')

        # metrics
        self.train_rmse = penne.metrics.WRMSE()
        self.train_rpa = penne.metrics.RPA()
        self.train_rca = penne.metrics.RCA()
        self.val_rmse = penne.metrics.WRMSE()
        self.val_rpa = penne.metrics.RPA()
        self.val_rca = penne.metrics.RCA()

        in_channels = [1, 1024, 128, 128, 128, 256]
        out_channels = [1024, 128, 128, 128, 256, 512]
        self.in_features = 2048

        # Shared layer parameters
        kernel_sizes = [512] + 5 * [64]
        strides = [4] + 5 * [1]

        # Overload with eps and momentum conversion given by MMdnn
        batch_norm_fn = functools.partial(torch.nn.BatchNorm1d,
                                          eps=self.epsilon,
                                          momentum=self.momentum)

        # Layer definitions
        self.conv = torch.nn.ModuleList()
        self.conv_BN = torch.nn.ModuleList()
        for i in range(6):
            self.conv.append(torch.nn.Conv1d(
                in_channels=in_channels[i],
                out_channels=out_channels[i],
                kernel_size=kernel_sizes[i],
                stride=strides[i])
            )
            self.conv_BN.append(batch_norm_fn(
                num_features=out_channels[i]
            ))

        self.classifier = torch.nn.Linear(
            in_features=self.in_features,
            out_features=penne.PITCH_BINS)

    ###########################################################################
    # Forward pass
    ###########################################################################

    def forward(self, x, embed=False):
        """Perform model inference"""
        x = self.embed(x)

        if embed:
            return x

        # Forward pass through layer six
        x = self.layer(x, self.conv[5], self.conv_BN[5], (31, 32))

        # shape=(batch, self.in_features)
        x = x.permute(0, 2, 1).reshape(-1, self.in_features)

        # Compute logits
        return self.classifier(x)

    ###########################################################################
    # Forward pass utilities
    ###########################################################################

    def embed(self, x):
        """Map input audio to pitch embedding"""
        # shape=(batch, 1, 1024)
        x = x[:, None, :]
        # Forward pass through first five layers
        for i in range(5):
            padding = (254, 254) if i == 0 else (31, 32)
            x = self.layer(x, self.conv[i], self.conv_BN[i], padding)
        return x

    def layer(self, x, conv, batch_norm, padding):
        """Forward pass through one layer"""
        x = F.pad(x, padding)
        x = conv(x)
        x = F.relu(x)
        x = batch_norm(x)
        x = F.max_pool1d(x, 2, 2)
        return F.dropout(x, p=0.25, training=self.training)


class PDCModel(torch.nn.Module):
    """PyTorch Lightning model definition"""

    def __init__(self, name='default'):
        super().__init__()

        self.epsilon = 0.0010000000474974513
        self.learning_rate = penne.LEARNING_RATE

        # equivalent to Keras default momentum
        self.momentum = 0.01

        self.name = name
        self.ex_batch = None
        self.best_loss = float('inf')

        # metrics
        self.train_rmse = penne.metrics.WRMSE()
        self.train_rpa = penne.metrics.RPA()
        self.train_rca = penne.metrics.RCA()
        self.val_rmse = penne.metrics.WRMSE()
        self.val_rpa = penne.metrics.RPA()
        self.val_rca = penne.metrics.RCA()

        in_channels = [1, 1024, 128, 128, 128, 256]
        out_channels = [1024, 128, 128, 128, 256, 512]
        self.in_features = 2048

        # Shared layer parameters
        kernel_sizes = [15] + 5 * [15]
        strides = [4] + 5 * [1]
        
        # Overload with eps and momentum conversion given by MMdnn
        batch_norm_fn = functools.partial(torch.nn.BatchNorm1d,
                                          eps=self.epsilon,
                                          momentum=self.momentum)

        # Layer definitions
        self.conv = torch.nn.ModuleList()
        self.conv_BN = torch.nn.ModuleList()
        for i in range(6):
            self.conv.append(PrimeDilatedConvolutionBlock(
                in_channels=in_channels[i],
                out_channels=out_channels[i],
                kernel_size=kernel_sizes[i],
                stride=strides[i]
            ))
            self.conv_BN.append(batch_norm_fn(
                num_features=out_channels[i]
            ))

        self.classifier = torch.nn.Linear(
            in_features=self.in_features,
            out_features=penne.PITCH_BINS)

    ###########################################################################
    # Forward pass
    ###########################################################################

    def forward(self, x, embed=False):
        """Perform model inference"""
        x = self.embed(x)

        if embed:
            return x

        # Forward pass through layer six
        x = self.layer(x, self.conv[5], self.conv_BN[5])

        # shape=(batch, self.in_features)
        x = x.reshape(-1, self.in_features)

        # Compute logits
        return self.classifier(x)

    ###########################################################################
    # Forward pass utilities
    ###########################################################################

    def embed(self, x):
        """Map input audio to pitch embedding"""
        # shape=(batch, 1, 1024)
        x = x[:, None, :]
        # Forward pass through first five layers
        for i in range(5):
            x = self.layer(x, self.conv[i], self.conv_BN[i])
        return x

    def layer(self, x, conv, batch_norm, padding=(0, 0, 0, 0), pooling=2):
        """Forward pass through one layer"""
        x = F.pad(x, padding)
        x = conv(x)
        x = F.relu(x)
        x = batch_norm(x)
        x = F.max_pool1d(x, pooling)
        return F.dropout(x, p=0.25, training=self.training)


# List of prime numbers for dilations
# And yes, 1 is technically not a prime number
PRIMES = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]


###############################################################################
# Constants
###############################################################################


class PrimeDilatedConvolutionBlock(torch.nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        primes=4):
        super().__init__()

        # Require output_channels to be evenly divisble by number of primes
        if out_channels // primes * primes != out_channels:
            raise ValueError(
                'output_channels must be evenly divisble by number of primes')

        # Initialize layers
        if stride == 1:
            conv_fn = functools.partial(
            torch.nn.Conv1d,
            kernel_size=kernel_size,
            padding='same')
        else:
            conv_fn = functools.partial(
            torch.nn.Conv1d,
            kernel_size=kernel_size,
            stride=stride)
        self.layers = torch.nn.ModuleList([
            conv_fn(in_channels, out_channels // primes, dilation=prime)
            for prime in PRIMES[:primes]])
        
        self.stride = stride

    def forward(self, x):
        """Forward pass"""
        # Forward pass through each
        if self.stride == 1:
            return torch.cat(tuple(layer(x) for layer in self.layers), dim=1)
        else:
            # have to handle padding when stride != 1 because padding='same' only works for stride == 1
            layer_outs = []
            for i, layer in enumerate(self.layers):
                dilation = PRIMES[i]
                L_in = x.shape[2]
                L_out = L_in // self.stride
                padding = math.ceil((self.stride * (L_out - 1) - L_in + dilation * (layer.kernel_size[0] - 1) + 1) / 2)
                layer_out = layer(F.pad(x, (padding, padding)))
                layer_outs.append(layer_out)
            return torch.cat(layer_outs, dim=1)

############################
# From harmof0: layers.py
############################

# Multiple Rate Dilated Convolution
class MRDConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dilation_list = [0, 12, 19, 24, 28, 31, 34, 36]):
        super().__init__()
        self.dilation_list = dilation_list
        self.conv_list = []
        for i in range(len(dilation_list)):
            self.conv_list += [torch.nn.Conv2d(in_channels, out_channels, kernel_size = [1, 1])]
        self.conv_list = torch.nn.ModuleList(self.conv_list)
        
    def forward(self, specgram):
        # input [b x C x T x n_freq]
        # output: [b x C x T x n_freq] 
        specgram
        dilation = self.dilation_list[0]
        y = self.conv_list[0](specgram)
        y = F.pad(y, pad=[0, dilation])
        y = y[:, :, :, dilation:]
        for i in range(1, len(self.conv_list)):
            dilation = self.dilation_list[i]
            x = self.conv_list[i](specgram)
            # => [b x T x (n_freq + dilation)]
            # x = F.pad(x, pad=[0, dilation])
            x = x[:, :, :, dilation:]
            n_freq = x.size()[3]
            y[:, :, :, :n_freq] += x

        return y

# Fixed Rate Dilated Casual Convolution
class FRDConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=[1,3], dilation=[1, 1]) -> None:
        super().__init__()
        right = (kernel_size[1]-1) * dilation[1]
        bottom = (kernel_size[0]-1) * dilation[0]
        self.padding = torch.nn.ZeroPad2d([0, right, 0 , bottom])
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self,x):
        x = self.padding(x)
        x = self.conv2d(x)
        return x


class WaveformToLogSpecgram(torch.nn.Module):
    def __init__(self, sample_rate, n_fft, fmin, bins_per_octave, freq_bins, hop_length, logspecgram_type): #, device
        super().__init__()

        e = freq_bins/bins_per_octave
        fmax = fmin * (2 ** e)

        self.logspecgram_type = logspecgram_type
        self.n_fft = n_fft
        hamming_window = torch.hann_window(self.n_fft)#.to(device)
        # => [1 x 1 x n_fft]
        hamming_window = hamming_window[None, None, :]
        self.register_buffer("hamming_window", hamming_window, persistent=False)

        # torch.hann_window()

        fre_resolution = sample_rate/n_fft

        idxs = torch.arange(0, freq_bins) #, device=device

        log_idxs = fmin * (2**(idxs/bins_per_octave)) / fre_resolution

        # Linear interpolation： y_k = y_i * (k-i) + y_{i+1} * ((i+1)-k)
        log_idxs_floor = torch.floor(log_idxs).long()
        log_idxs_floor_w = (log_idxs - log_idxs_floor).reshape([1, 1, freq_bins])
        log_idxs_ceiling = torch.ceil(log_idxs).long()
        log_idxs_ceiling_w = (log_idxs_ceiling - log_idxs).reshape([1, 1, freq_bins])
        self.register_buffer("log_idxs_floor", log_idxs_floor, persistent=False)
        self.register_buffer("log_idxs_floor_w", log_idxs_floor_w, persistent=False)
        self.register_buffer("log_idxs_ceiling", log_idxs_ceiling, persistent=False)
        self.register_buffer("log_idxs_ceiling_w", log_idxs_ceiling_w, persistent=False)

        self.waveform_to_specgram = torchaudio.transforms.Spectrogram(n_fft, hop_length=hop_length)#.to(device)

        assert(bins_per_octave % 12 == 0)
        bins_per_semitone = bins_per_octave // 12
        
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)

    def forward(self, waveforms):
        # inputs: [b x num_frames x frame_len]
        # outputs: [b x num_frames x n_bins]

        if(self.logspecgram_type == 'logharmgram'):
            waveforms = waveforms * self.hamming_window
            specgram =  torch.fft.fft(waveforms)
            specgram = torch.abs(specgram[:, :, :self.n_fft//2 + 1])
            specgram = specgram * specgram
            # => [num_frames x n_fft//2 x 1]
            # specgram = torch.unsqueeze(specgram, dim=2)

            # => [b x freq_bins x T]
            specgram = specgram[:,:, self.log_idxs_floor] * self.log_idxs_floor_w + specgram[:, :, self.log_idxs_ceiling] * self.log_idxs_ceiling_w

        specgram_db = self.amplitude_to_db(specgram)
        # specgram_db = specgram_db[:, :, :-1] # remove the last frame.
        # specgram_db = specgram_db.permute([0, 2, 1])
        return specgram_db

############################
# From harmof0: network.py
############################

def dila_conv_block( 
    in_channel, out_channel, 
    bins_per_octave,
    n_har,
    dilation_mode,
    dilation_rate,
    dil_kernel_size,
    kernel_size = [1,3],
    padding = [0,1],
):

    conv = torch.nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding)
    batch_norm = torch.nn.BatchNorm2d(out_channel)

    # dilation mode: 'log_scale', 'fixed'
    if(dilation_mode == 'log_scale'):
        a = np.log(np.arange(1, n_har + 1))/np.log(2**(1.0/bins_per_octave))
        dilation_list = a.round().astype(np.int)
        conv_log_dil = MRDConv(out_channel, out_channel, dilation_list)
        return torch.nn.Sequential(
            conv,torch.nn.ReLU(),
            conv_log_dil,torch.nn.ReLU(),
            batch_norm,
            # pool
        )
    elif(dilation_mode == 'fixed_causal'):
        dilation_list = np.array([i * dil_kernel_size[1] for i in range(dil_kernel_size[1])])
        causal_conv = FRDConv(out_channel, out_channel, dil_kernel_size, dilation=[1, dilation_rate])
        return torch.nn.Sequential(
            conv,torch.nn.ReLU(),
            causal_conv,torch.nn.ReLU(),
            batch_norm,
            # pool
        )
    elif(dilation_mode == 'fixed'):
        conv_dil = torch.nn.Conv2d(out_channel, out_channel, kernel_size=dil_kernel_size, padding='same', dilation=[1, dilation_rate])
        
        return torch.nn.Sequential(
            conv,torch.nn.ReLU(),
            conv_dil,torch.nn.ReLU(),
            batch_norm,
            # pool
        )
    else:
        assert False, "unknown dilation type: " + dilation_mode


class HarmoF0(torch.nn.Module):
    def __init__(self, 
            sample_rate=16000, 
            n_freq=512, 
            n_har=12, 
            bins_per_octave=12 * 4, 
            dilation_modes=['log_scale', 'fixed', 'fixed', 'fixed'],
            dilation_rates=[48, 48, 48, 48],
            logspecgram_type='logharmgram',
            channels=[32, 64, 128, 128],
            fmin=27.5,
            freq_bins=360,
            dil_kernel_sizes= [[1, 3], [1,3], [1,3], [1,3]],
        ):
        super().__init__()
        self.logspecgram_type = logspecgram_type

        n_fft = n_freq * 2
        self.n_freq = n_freq
        self.freq_bins = freq_bins
        
        self.waveform_to_logspecgram = WaveformToLogSpecgram(sample_rate, n_fft, fmin, bins_per_octave, freq_bins, n_freq, logspecgram_type) #, device

        bins = bins_per_octave

        # [b x 1 x T x 88*8] => [b x 32 x T x 88*4]
        self.block_1 = dila_conv_block(1, channels[0], bins, n_har=n_har, dilation_mode=dilation_modes[0], dilation_rate=dilation_rates[0], dil_kernel_size=dil_kernel_sizes[0], kernel_size=[3, 3], padding=[1,1])
        
        bins = bins // 2
        # => [b x 64 x T x 88*4]
        self.block_2 = dila_conv_block(channels[0], channels[1], bins, 3, dilation_mode=dilation_modes[1], dilation_rate=dilation_rates[1], dil_kernel_size=dil_kernel_sizes[1], kernel_size=[3, 3], padding=[1,1])
        # => [b x 128 x T x 88*4]
        self.block_3 = dila_conv_block(channels[1], channels[2], bins, 3, dilation_mode=dilation_modes[2], dilation_rate=dilation_rates[2], dil_kernel_size=dil_kernel_sizes[2], kernel_size=[3, 3], padding=[1,1])
        # => [b x 128 x T x 88*4]
        self.block_4 = dila_conv_block(channels[2], channels[3], bins, 3, dilation_mode=dilation_modes[3], dilation_rate=dilation_rates[3], dil_kernel_size=dil_kernel_sizes[3], kernel_size=[3, 3], padding=[1,1])

        self.conv_5 = torch.nn.Conv2d(channels[3], channels[3]//2, kernel_size=[1,1])
        self.conv_6 = torch.nn.Conv2d(channels[3]//2, 1, kernel_size=[1,1])

    def forward(self, waveforms):
        # input: [b x num_frames x frame_len]
        # output: [b x num_frames x 360]

        specgram = self.waveform_to_logspecgram(waveforms.transpose(2,1)).float()
        # => [b x 1 x num_frames x n_bins]
        x = specgram[None, :]

        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)

        # [b x 128 x T x 360] => [b x 64 x T x 360]
        x = self.conv_5(x)
        x = torch.relu(x)
        x = self.conv_6(x)
        x = torch.sigmoid(x)

        x = torch.squeeze(x, dim=1)
        # x = torch.clip(x, 1e-4, 1 - 1e-4)
        # => [num_frames x n_bins]
        return x, specgram