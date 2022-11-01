import torch

import penne


###############################################################################
# DeepF0 model implementation
###############################################################################


class Deepf0(torch.nn.Sequential):

    def __init__(self, channels=128, kernel_size=64):
        super().__init__(
            CausalConv1d(1, channels, kernel_size),
            Block(channels, channels, kernel_size, 1),
            Block(channels, channels, kernel_size, 2),
            Block(channels, channels, kernel_size, 4),
            Block(channels, channels, kernel_size, 8),
            torch.nn.AvgPool1d(kernel_size),
            Flatten(),
            torch.nn.Linear(2048, penne.PITCH_BINS))

    def forward(self, frames):
        # shape=(batch, 1, penne.WINDOW_SIZE) =>
        # shape=(batch, penne.PITCH_BINS, penne.NUM_TRAINING_FRAMES)
        return super().forward(frames)[:, :, None]


###############################################################################
# Utilities
###############################################################################


class Block(torch.nn.Sequential):

    def __init__(self, input_channels, output_channels, kernel_size, dilation):
        super().__init__(
            CausalConv1d(
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


class CausalConv1d(torch.nn.Conv1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True):
        self.pad = (kernel_size - 1) * dilation
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.pad,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, input):
        result = super().forward(input)
        if self.pad != 0:
            return result[:, :, :-self.pad]
        return result


class Flatten(torch.nn.Module):

    def forward(self, x):
        return x.reshape(x.shape[0], -1)
