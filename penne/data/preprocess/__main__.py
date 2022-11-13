import argparse

import penne


###############################################################################
# Preprocess datasets
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Preprocess datasets')
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['mdb', 'ptdb'],
        help='The datasets to preprocess')
    return parser.parse_args()


penne.preprocess.datasets(**vars(parse_args()))
