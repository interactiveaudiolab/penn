import torch

import penne


###############################################################################
# Crepe
###############################################################################


class Crepe(torch.nn.Sequential):

    def __init__(self):
        super().__init__()
        in_channels = [1, 1024, 128, 128, 128, 256]
        out_channels = [1024, 128, 128, 128, 256, 512]
        kernel_sizes = [512] + 5 * [64]
        strides = [4] + 5 * [1]
        if penne.MAX_POOL is None:
            strides = [2 * stride for stride in strides]
        padding = [(254, 254)] + 5 * [(31, 32)]
        super().__init__(*(
            (
                Block(i, o, k, s, p) for i, o, k, s, p in
                zip(in_channels, out_channels, kernel_sizes, strides, padding)
            ) +
            (
                Flatten(),
                torch.nn.Linear(
                    in_features=2048,
                    out_features=penne.PITCH_BINS)
            )
        ))

    def forward(self, audio):
        # shape=(batch, 1, penne.WINDOW_SIZE) =>
        # shape=(batch, penne.PITCH_BINS, penne.NUM_TRAINING_FRAMES)
        return super().forward()[:, :, None]


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
        padding):
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
            layers += (torch.nn.InstanceNorm1d(out_channels))

        # Maybe add max pooling
        if penne.MAX_POOL is not None:
            layers += (torch.nn.MaxPool1d(*penne.MAX_POOL),)

        # Maybe add dropout
        if penne.DROPOUT is not None:
            layers += (torch.nn.Dropout(penne.DROPOUT),)

        super().__init__(*layers)


class Flatten(torch.nn.Module):

    def forward(self, x):
        return x.reshape(x.shape[0], -1)
