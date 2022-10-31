import torch

import penne


###############################################################################
# DeepF0 model implementation
###############################################################################


class Deepf0(torch.nn.Sequential):

    def __init__(self, channels, kernel_size):
        super().__init__(
            torch.nn.Conv1d(1, channels, kernel_size),
            Block(channels, channels, kernel_size, 1),
            Block(channels, channels, kernel_size, 2),
            Block(channels, channels, kernel_size, 4),
            Block(channels, channels, kernel_size, 8),
            torch.nn.AvdPool1d(kernel_size),
            Flatten(),
            torch.nn.Linear(2048, penne.PITCH_BINS))


###############################################################################
# Utilities
###############################################################################


class Block(torch.nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, dilation):
        super().__init__(
            torch.nn.Conv1d(
                input_channels,
                output_channels,
                kernel_size,
                dilation=dilation),
            torch.nn.ReLU(),
            torch.nn.utils.weight_norm(torch.nn.Conv1d(
                output_channels,
                output_channels,
                1)))

    def forward(self, x):
        return torch.nn.functional.relu(super().forward(x) + x)


class Flatten(torch.nn.Module):

    def forward(self, x):
        return x.reshape(x.shape[0], -1)
