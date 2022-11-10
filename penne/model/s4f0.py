import functools
import math

import einops
import torch
import opt_einsum

import penne


###############################################################################
# Crepe
###############################################################################


class S4f0(torch.nn.Sequential):

    def __init__(self, kernel_size=5):
        conv_fn = functools.partial(
            torch.nn.Conv1d,
            kernel_size=kernel_size,
            padding='same')
        super().__init__(
            conv_fn(1, penne.NUM_TRAINING_SAMPLES),
            StructuredGlobalConvolution(
                penne.NUM_TRAINING_SAMPLES,
                penne.NUM_TRAINING_SAMPLES),
            StructuredGlobalConvolution(
                penne.NUM_TRAINING_SAMPLES,
                penne.NUM_TRAINING_SAMPLES),
            Block(
                penne.NUM_TRAINING_SAMPLES,
                penne.NUM_TRAINING_SAMPLES // 4,
                kernel_size,
                1,
                kernel_size // 2,
                (4, 4),
                penne.NUM_TRAINING_SAMPLES),
            Block(
                penne.NUM_TRAINING_SAMPLES // 4,
                penne.NUM_TRAINING_SAMPLES // 16,
                kernel_size,
                1,
                kernel_size // 2,
                (4, 4),
                penne.NUM_TRAINING_SAMPLES // 4),
            Block(
                penne.NUM_TRAINING_SAMPLES // 16,
                penne.NUM_TRAINING_SAMPLES // 16,
                kernel_size,
                1,
                kernel_size // 2,
                (2, 2),
                penne.NUM_TRAINING_SAMPLES // 16),
            Flatten(),
            torch.nn.Linear(
                in_features=2048,
                out_features=penne.PITCH_BINS))

    def forward(self, frames):
        # shape=(batch, 1, penne.WINDOW_SIZE) =>
        # shape=(batch, penne.PITCH_BINS, penne.NUM_TRAINING_FRAMES)
        return super().forward(frames)[:, :, None]


###############################################################################
# Utilities
###############################################################################


class Block(torch.nn.Sequential):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        pooling,
        length):
        layers = (
            torch.nn.ConstantPad1d(padding, 0),
            torch.nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride),
            torch.nn.ReLU())

        # Maybe add normalization
        if penne.NORMALIZATION == 'batch':
            layers += (torch.nn.BatchNorm1d(out_channels, momentum=.01),)
        elif penne.NORMALIZATION == 'instance':
            layers += (torch.nn.InstanceNorm1d(out_channels),)
        elif penne.NORMALIZATION == 'layer':
            layers += (torch.nn.LayerNorm((out_channels, length)),)

        # Add max pooling
        layers += (torch.nn.MaxPool1d(*pooling),)

        # Maybe add dropout
        if penne.DROPOUT is not None:
            layers += (torch.nn.Dropout(penne.DROPOUT),)

        super().__init__(*layers)


class Flatten(torch.nn.Module):

    def forward(self, x):
        return x.reshape(x.shape[0], -1)


class StructuredGlobalConvolution(torch.nn.Module):

    def __init__(
            self,
            channels,
            length,
            kernel_size=64,
            bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.D = torch.nn.Parameter(torch.randn(1, channels))
        self.linear = TransposedLinear(channels, channels)
        self.num_scales = 1 + math.ceil(math.log2(length / kernel_size))
        self.kernel_list = torch.nn.ParameterList()
        for _ in range(self.num_scales):
            kernel = torch.nn.Parameter(torch.randn(1 + bidirectional, channels, kernel_size))
            kernel._optim = {'lr': .001}
            self.kernel_list.append(kernel)
        self.register_buffer(
            'multiplier',
            torch.linspace(1, 4, channels).view(1, -1, 1))
        self.register_buffer('kernel_norm', torch.ones(channels, 1))
        self.register_buffer('kernel_norm_initialized', torch.tensor(0, dtype=torch.bool))

    def forward(self, u):
        L = u.size(-1)

        kernel_list = []
        for i in range(self.num_scales):
            kernel = torch.nn.functional.interpolate(
                self.kernel_list[i],
                scale_factor=2 ** (max(0, i - 1)),
                mode='linear',
            ) * self.multiplier ** (self.num_scales - i - 1)
            kernel_list.append(kernel)
        k = torch.cat(kernel_list, dim=-1)

        if not self.kernel_norm_initialized:
            self.kernel_norm = k.norm(dim=-1, keepdim=True).detach()
            self.kernel_norm_initialized = torch.tensor(
                1,
                dtype=torch.bool,
                device=k.device)

        if k.size(-1) > L:
            k = k[..., :L]
        elif k.size(-1) < L:
            k = torch.nn.functional.pad(k, (0, L - k.size(-1)))

        k = k / self.kernel_norm #* (L / self.l_max) ** 0.5

        # Convolution
        if self.bidirectional:
            k0, k1 = einops.rearrange(k, '(s c) h l -> s c h l', s=2)
            k = torch.nn.functional.pad(k0, (0, L)) \
                    + torch.nn.functional.pad(k1.flip(-1), (L, 0))

        k_f = torch.fft.rfft(k, n=2 * L) # (C H L)
        u_f = torch.fft.rfft(u, n=2 * L) # (B H L)
        y_f = opt_einsum.contract('bhl,chl->bchl', u_f, k_f) # k_f.unsqueeze(-4) * u_f.unsqueeze(-3) # (B C H L)
        y = torch.fft.irfft(y_f, n=2 * L)[..., :L] # (B C H L)

        # Compute D term in state space equation - essentially a skip connection
        y = (y + opt_einsum.contract('bhl,ch->bchl', u, self.D))[:, 0]

        y = torch.nn.functional.gelu(y)

        y = torch.nn.functional.gelu(self.linear(y))

        return y


class TransposedLinear(torch.nn.Module):

    def __init__(self, d_input, d_output):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(d_output, d_input))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5)) # torch.nn.Linear default init
        self.bias = torch.nn.Parameter(torch.empty(d_output))
        bound = 1 / math.sqrt(d_input)
        torch.nn.init.uniform_(self.bias, -bound, bound)
        setattr(self.bias, "_optim", {"weight_decay": 0.0})

    def forward(self, x):
        return (
            opt_einsum.contract('b u ..., v u -> b v ...', x, self.weight) +
            self.bias.view(-1, *[1] * len(x.shape[2:])))
