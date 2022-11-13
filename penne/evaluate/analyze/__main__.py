import argparse
from pathlib import Path

import penne


###############################################################################
# Analyze results
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Analyze evaluation results')
    parser.add_argument(
        '--runs',
        nargs='+',
        help='The runs to analyze. Defaults to all in eval directory.')
    return parser.parse_args()


if __name__ == '__main__':
    penne.evaluate.analyze()
