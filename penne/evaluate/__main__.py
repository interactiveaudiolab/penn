import argparse
from pathlib import Path

import penne


###############################################################################
# Evaluate pitch and periodicity estimation
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datasets',
        help='The datasets to evaluate')
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=penne.DEFAULT_CHECKPOINT,
        help='The checkpoint file to evaluate')
    parser.add_argument(
        '--gpu',
        type=int,
        help='The index of the GPU to use for evaluation')

    return parser.parse_args()


penne.evaluate.datasets(**vars(parse_args()))
