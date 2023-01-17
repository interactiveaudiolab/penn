import argparse

import penn
from pathlib import Path


###############################################################################
# Entry point
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--files',
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
        type=float,
        default=penn.HOPSIZE_SECONDS,
        help=(
            'The hopsize in seconds. '
            f'Defaults to {penn.HOPSIZE_SECONDS} seconds.'))
    parser.add_argument(
        '--fmin',
        type=float,
        default=penn.FMIN,
        help=(
            'The minimum frequency allowed in Hz. '
            f'Defaults to {penn.FMIN} Hz.'))
    parser.add_argument(
        '--fmax',
        type=float,
        default=penn.FMAX,
        help=(
            'The maximum frequency allowed in Hz. '
            f'Defaults to {penn.FMAX} Hz.'))
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=penn.DEFAULT_CHECKPOINT,
        help=(
            'The model checkpoint file. '
            f'Defaults to {penn.DEFAULT_CHECKPOINT}'))
    parser.add_argument(
        '--batch_size',
        type=int,
        default=penn.EVALUATION_BATCH_SIZE,
        help=(
            'The number of frames per batch. '
            f'Defaults to {penn.EVALUATION_BATCH_SIZE}.'))
    parser.add_argument(
        '--pad',
        action='store_true',
        help=(
            'If true, centers frames at '
            'hopsize / 2, 3 * hopsize / 2, 5 * ...'))
    parser.add_argument(
        '--interp_unvoiced_at',
        type=float,
        default=penn.DEFAULT_VOICING_THRESHOLD,
        help='Specifies voicing threshold for interpolation')
    parser.add_argument(
        '--gpu',
        type=int,
        help='The index of the gpu to perform inference on. Defaults to CPU.')

    return parser.parse_known_args()[0]


penn.from_files_to_files(**vars(parse_args()))
