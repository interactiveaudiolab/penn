import torch

import penn


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
            ([penn.model.Normalize()] if penn.NORMALIZE_INPUT else []) +
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
                penn.model.Flatten(),
                torch.nn.Linear(
                    in_features=2048,
                    out_features=penn.PITCH_BINS)
            ]
        ))

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
        if penn.NORMALIZATION == 'batch':
            layers += (torch.nn.BatchNorm1d(out_channels, momentum=.01),)
        elif penn.NORMALIZATION == 'instance':
            layers += (torch.nn.InstanceNorm1d(out_channels),)
        elif penn.NORMALIZATION == 'layer':
            layers += (torch.nn.LayerNorm((out_channels, length)),)
        else:
            raise ValueError(
                f'Normalization method {penn.NORMALIZATION} is not defined')

        # Add max pooling
        layers += (torch.nn.MaxPool1d(2, 2),)

        # Maybe add dropout
        if penn.DROPOUT is not None:
            layers += (torch.nn.Dropout(penn.DROPOUT),)

        super().__init__(*layers)
