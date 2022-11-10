import torch

import penne


class Fcnf0(torch.nn.Sequential):

    def __init__(self):
        # TODO - layer norm
        super().__init__(
            Block(1, 256, 481, (2, 2)),
            Block(256, 32, 225, (2, 2)),
            Block(32, 32, 97, (2, 2)),
            Block(32, 128, 66),
            Block(128, 256, 35),
            Block(256, 512, 4),
            torch.nn.Conv1d(512, penne.PITCH_BINS, 4))

    def forward(self, frames):
        return super().forward(frames[:, :, 16:-15]).permute(2, 1, 0)


class Block(torch.nn.Sequential):
    # TODO - lengths

    def __init__(
        self,
        in_channels,
        out_channels,
        length=1,
        pooling=None,
        kernel_size=32):
        layers = (
            torch.nn.Conv1d(in_channels, out_channels, kernel_size),
            torch.nn.ReLU())

        # Maybe add pooling
        if pooling is not None:
            layers += (torch.nn.MaxPool1d(*pooling),)

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

        # Maybe add dropout
        if penne.DROPOUT is not None:
            layers += (torch.nn.Dropout(penne.DROPOUT),)

        super().__init__(*layers)
