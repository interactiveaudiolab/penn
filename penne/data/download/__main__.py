import argparse

import penne


###############################################################################
# Download datasets
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Download datasets')
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['mdb', 'ptdb'],
        help='The datasets to download')
    return parser.parse_args()


penne.data.download.datasets(**vars(parse_args()))
