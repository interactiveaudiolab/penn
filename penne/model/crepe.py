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
        kernels = [512] + 5 * [64]
        strides = [4] + 5 * [1]
        padding = [(254, 254)] + 5 * [(31, 32)]
        lengths = [256, 128, 64, 32, 16, 8]
        super().__init__(*(
            ([penne.model.Normalize()] if penne.NORMALIZE_INPUT else []) +
            [
                Block(i, o, k, s, p, l) for i, o, k, s, p, l in
                zip(
                    in_channels,
                    out_channels,
                    kernels,
                    strides,
                    padding,
                    lengths
                )
            ] +
            [
                penne.model.Flatten(),
                torch.nn.Linear(
                    in_features=2048,
                    out_features=penne.PITCH_BINS)
            ]
        ))

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
        else:
            raise ValueError(
                f'Normalization method {penne.NORMALIZATION} is not defined')

        # Add max pooling
        layers += (torch.nn.MaxPool1d(2, 2),)

        # Maybe add dropout
        if penne.DROPOUT is not None:
            layers += (torch.nn.Dropout(penne.DROPOUT),)

        super().__init__(*layers)
