import torch

import penn


###############################################################################
# Voiced/unvoiced
###############################################################################


def interpolate(pitch, periodicity, value):
    """Fill unvoiced regions via linear interpolation"""
    # Threshold periodicity
    voiced = threshold(periodicity, value)

    # Pitch is linear in base-2 log-space
    pitch = torch.log2(pitch)

    # Interpolate
    pitch[~voiced] = penn.interpolate(
        pitch,
        torch.where(~voiced),
        torch.where(voiced))

    return 2 ** pitch, voiced


def threshold(periodicity, value):
    """Threshold periodicity to produce voiced/unvoiced classifications"""
    return periodicity > value
