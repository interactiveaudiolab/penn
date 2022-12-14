import argparse
from pathlib import Path

import penn


###############################################################################
# Create figure
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Create logits figure')
    parser.add_argument(
        '--audio_file',
        required=True,
        type=Path,
        help='The audio file to plot the logits of')
    parser.add_argument(
        '--output_file',
        required=True,
        type=Path,
        help='The jpg file to save the plot')
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=penn.DEFAULT_CHECKPOINT,
        help='The checkpoint file to use for inference')
    parser.add_argument(
        '--gpu',
        type=int,
        help='The index of the GPU to use for inference')
    return parser.parse_known_args()[0]


penn.plot.logits.from_file_to_file(**vars(parse_args()))
