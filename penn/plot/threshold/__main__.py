import argparse
from pathlib import Path

import penn


###############################################################################
# Periodicity threshold figure
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Create periodicity threshold figure')
    parser.add_argument(
        '--names',
        required=True,
        nargs='+',
        help='Corresponding labels for each evaluation')
    parser.add_argument(
        '--evaluations',
        type=Path,
        required=True,
        nargs='+',
        help='The evaluations to plot')
    parser.add_argument(
        '--output_file',
        type=Path,
        required=True,
        help='The output jpg file')
    return parser.parse_known_args()[0]


penn.plot.threshold.from_evaluations(**vars(parse_args()))
