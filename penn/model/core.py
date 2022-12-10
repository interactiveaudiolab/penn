import torch

import penn


class Flatten(torch.nn.Module):

    def forward(self, x):
        return x.reshape(x.shape[0], -1)


class Normalize(torch.nn.Module):

    def forward(self, frames):
        return penn.normalize(frames)
