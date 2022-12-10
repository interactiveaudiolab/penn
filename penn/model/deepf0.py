import torch

import penn


###############################################################################
# DeepF0 model implementation
###############################################################################


class Deepf0(torch.nn.Sequential):

    def __init__(self, channels=128, kernel_size=64):
        layers = (penn.model.Normalize(),) if penn.NORMALIZE_INPUT else ()
        layers += (
            CausalConv1d(1, channels, kernel_size),
            Block(channels, channels, kernel_size, 1),
            Block(channels, channels, kernel_size, 2),
            Block(channels, channels, kernel_size, 4),
            Block(channels, channels, kernel_size, 8),
            torch.nn.AvgPool1d(kernel_size),
            penn.model.Flatten(),
            torch.nn.Linear(2048, penn.PITCH_BINS))
        super().__init__(*layers)

    def forward(self, frames):
        # shape=(batch, 1, penn.WINDOW_SIZE) =>
        # shape=(batch, penn.PITCH_BINS, penn.NUM_TRAINING_FRAMES)
        return super().forward(frames)[:, :, None]


###############################################################################
# Utilities
###############################################################################


class Block(torch.nn.Sequential):

    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size,
        dilation):
        if penn.NORMALIZATION == 'weight':
            norm_conv = [
                torch.nn.utils.weight_norm(
                    torch.nn.Conv1d(output_channels, output_channels, 1))]
        elif penn.NORMALIZATION == 'layer':
            norm_conv = [
                torch.nn.LayerNorm(
                    (output_channels, penn.NUM_TRAINING_SAMPLES)),
                torch.nn.Conv1d(output_channels, output_channels, 1)]
        else:
            raise ValueError(
                f'Normalization method {penn.NORMALIZATION} is not defined')

        super().__init__(*(
            [
                CausalConv1d(
                    input_channels,
                    output_channels,
                    kernel_size,
                    dilation=dilation),
                torch.nn.ReLU()
            ] + norm_conv))

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
