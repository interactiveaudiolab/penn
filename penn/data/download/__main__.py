import argparse

import penn


###############################################################################
# Download datasets
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Download datasets')
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=penn.DATASETS,
        help='The datasets to download')
    return parser.parse_args()


penn.data.download.datasets(**vars(parse_args()))
