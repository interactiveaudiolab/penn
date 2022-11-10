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
    parser.add_argument(
        '--output_directory',
        default=penne.RESULTS_DIR,
        type=Path,
        help='The directory to save results tables')
    return parser.parse_args()


if __name__ == '__main__':
    penne.evaluate.analyze()
