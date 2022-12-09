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
        help='The audio files to process')
    parser.add_argument(
        '--output_prefixes',
        nargs='+',
        type=Path,
        help=(
            'The files to save pitch and periodicity without extension. '
            'Defaults to audio_files without extensions.'))
    parser.add_argument(
        '--hopsize',
        type=int,
        default=penne.HOPSIZE_SECONDS,
        help=(
            'The hopsize in seconds. '
            f'Defaults to {penne.HOPSIZE_SECONDS} seconds.'))
    parser.add_argument(
        '--fmin',
        type=float,
        default=penne.FMIN,
        help=(
            'The minimum frequency allowed in Hz. '
            f'Defaults to {penne.FMIN} Hz.'))
    parser.add_argument(
        '--fmax',
        type=float,
        default=penne.FMAX,
        help=(
            'The maximum frequency allowed in Hz. '
            f'Defaults to {penne.FMAX} Hz.'))
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=penne.DEFAULT_CHECKPOINT,
        help=(
            'The model checkpoint file. '
            f'Defaults to {penne.DEFAULT_CHECKPOINT}'))
    parser.add_argument(
        '--batch_size',
        type=int,
        default=penne.EVALUATION_BATCH_SIZE,
        help=(
            'The number of frames per batch. '
            f'Defaults to {penne.EVALUATION_BATCH_SIZE}.'))
    parser.add_argument(
        '--gpu',
        type=int,
        help='The index of the gpu to perform inference on. Defaults to CPU.')

    return parser.parse_known_args()[0]


penne.from_files_to_files(**vars(parse_args()))
