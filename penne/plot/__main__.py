import argparse

import penne


###############################################################################
# Create figure
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Create figure')
    parser.add_argument(
        '--dataset',
        required=True,
        help='The dataset to draw an example from')
    parser.add_argument(
        '--stem',
        required=True,
        help='The example to draw from the dataset')
    return parser.parse_args()


penne.plot.from_file_to_file(**vars(parse_args()))
