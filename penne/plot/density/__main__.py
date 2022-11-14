import argparse
from pathlib import Path

import penne


###############################################################################
# Create figure
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Create density figure')
    parser.add_argument(
        '--true_datasets',
        nargs='+',
        required=True,
        help='Datasets to use for ground truth density')
    parser.add_argument(
        '--inference_datasets',
        nargs='+',
        required=True,
        help='Datasets to use for inference density')
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=penne.DEFAULT_CHECKPOINT,
        help='The checkpoint file to use for inference')
    parser.add_argument(
        '--gpu',
        type=int,
        help='The index of the GPU to use for inference')


penne.plot.density.from_file_to_file(**vars(parse_args()))
