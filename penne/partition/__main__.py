import argparse

import penne


###############################################################################
# Partition datasets
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Partition datasets')
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['mdb', 'ptdb'],
        help='The datasets to partition')
    return parser.parse_args()


penne.partition.datasets(**vars(parse_args()))
