import argparse
from pathlib import Path

import penne


###############################################################################
# Create figure
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Create inference figure')
    parser.add_argument(
        '--audio_file',
        required=True,
        type=Path,
        help='The dataset to draw an example from')
    parser.add_argument(
        '--output_file',
        type=Path,
        help='The output file. Defaults to audio_file with .jpg extension.')
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=penne.DEFAULT_CHECKPOINT,
        help='The checkpoint file to use for inference')
    parser.add_argument(
        '--gpu',
        type=int,
        help='The index of the GPU to use for inference')
    return parser.parse_args()


penne.plot.inference.from_file_to_file(**vars(parse_args()))
