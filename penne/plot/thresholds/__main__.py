import argparse
from pathlib import Path

import penne


###############################################################################
# Periodicity threshold figure
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Create periodicity threshold figure')
    parser.add_argument(
        '--output_file',
        type=Path,
        help='The output jpg file')
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=penne.DEFAULT_CHECKPOINT,
        help='The checkpoint file to use for inference')
    parser.add_argument(
        '--gpu',
        type=int,
        help='The index of the GPU to use for inference')
    return parser.parse_known_args()[0]


penne.plot.thresholds.from_file_to_file(**vars(parse_args()))
