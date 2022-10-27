import argparse

import penne
from pathlib import Path


###############################################################################
# Entry point
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--audio_files',
        nargs='+',
        required=True,
        type=Path,
        help='The audio file to process')
    parser.add_argument(
        '--output_prefixes',
        nargs='+',
        type=Path,
        help='The files to save pitch and periodicity without extension')
    parser.add_argument(
        '--hopsize',
        type=int,
        default=penne.HOPSIZE_SECONDS,
        help='The hopsize in seconds')
    parser.add_argument(
        '--fmin',
        type=float,
        help='The minimum frequency allowed')
    parser.add_argument(
        '--fmax',
        type=float,
        help='The maximum frequency allowed')
    parser.add_argument(
        '--model',
        default=penne.MODEL,
        help='The name of the estimator model')
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=penne.DEFAULT_CHECKPOINT,
        help='The model checkpoint file')
    parser.add_argument(
        '--batch_size',
        type=int,
        help='The number of frames per batch')
    parser.add_argument(
        '--gpu',
        type=int,
        help='The index of the gpu to perform inference on')

    return parser.parse_args()


penne.from_files_to_files(**vars(parse_args()))
