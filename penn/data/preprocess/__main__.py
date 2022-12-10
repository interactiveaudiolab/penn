import argparse

import penn


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
    return parser.parse_known_args()[0]


penn.data.preprocess.datasets(**vars(parse_args()))
