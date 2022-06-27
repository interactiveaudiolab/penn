import functools

import torch
import torch.nn.functional as F
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