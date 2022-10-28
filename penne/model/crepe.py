import torch

import penne


###############################################################################
# Crepe
###############################################################################


class Crepe(torch.nn.Module):
    """Crepe model definition"""

    def __init__(self):
        super().__init__()
        in_channels = [1, 1024, 128, 128, 128, 256]
        out_channels = [1024, 128, 128, 128, 256, 512]
        kernel_sizes = [512] + 5 * [64]
        strides = [4] + 5 * [1]
        if penne.MAX_POOL is None:
            strides = [4 * stride for stride in strides]
        padding = [(254, 254)] + 5 * [(31, 32)]
        self.blocks = torch.nn.Sequential(*(
            Block(i, o, k, s, p) for i, o, k, s, p in
            zip(in_channels, out_channels, kernel_sizes, strides, padding)))
        self.classifier = torch.nn.Linear(
            in_features=2048,
            out_features=penne.PITCH_BINS)

    def forward(self, audio):
        # Maybe normalize audio
        if penne.CREPE_NORMALIZE:

            # Mean-center
            audio -= audio.mean(dim=2, keepdim=True)

            # Scale
            audio /= torch.max(
                torch.tensor(1e-10, device=audio.device),
                audio.std(dim=2, keepdim=True))

        # Infer
        # shape=(batch, 1, penne.WINDOW_SIZE) =>
        # shape=(batch, penne.PITCH_BINS)
        return self.classifier(
            self.blocks(audio).reshape(audio.shape[0], 2048))[:, :, None]


###############################################################################
# Utilities
###############################################################################


class Block(torch.nn.Sequential):
    """Crepe block definition"""

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

        # Maybe add batch normalization
        if penne.NORMALIZATION == 'batch':
            layers += (torch.nn.BatchNorm1d(out_channels, momentum=.01),)

        # Maybe add max pooling
        if penne.MAX_POOL is not None:
            layers += (torch.nn.MaxPool1d(*penne.MAX_POOL),)

        # Maybe add dropout
        if penne.DROPOUT is not None:
            layers += (torch.nn.Dropout(penne.DROPOUT),)

        super().__init__(*layers)
