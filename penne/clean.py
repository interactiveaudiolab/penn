"""clean.py - clean out potentially problematic data"""


import argparse
import json
import random
import math
import os
from pathlib import Path

import penne


###############################################################################
# Clean
###############################################################################

def clean_data(results_path, partition, dataset, rpa_threshold, f1_threshold):
    # results_path should be the per stem output from evaluation
    with open(results_path) as f:
        results = json.load(f)
        # scores sorted by rpa
        scores = [(name, value['rpa'], value['f1']) for name, value in results.items()]
        # remove stems that don't meet score thresholds
        good_stems = [stem for stem, rpa, f1 in scores if rpa > rpa_threshold and f1 > f1_threshold]

        # update existing clean_partition unless we don't have a clean_partition yet
        if os.path.exists(penne.ASSETS_DIR / dataset / 'clean_partition.json'):
            with open(penne.ASSETS_DIR / dataset / 'clean_partition.json') as g:
                # replace partition with new stems
                parts = json.load(g)
                parts[partition] = good_stems
                json.dump(parts, g)
        else:
            with open(penne.ASSETS_DIR / dataset / 'partition.json') as g:
                # replace partition with new stems
                parts = json.load(g)
                parts[partition] = good_stems
                with open(penne.ASSETS_DIR / dataset / 'clean_partition.json', 'w') as h:
                    json.dump(parts, h)


###############################################################################
# Entry point
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'results',
        help='Path to per stem evaluation results json')
    parser.add_argument(
        'partition',
        help='Path to per stem evaluation results json')
    parser.add_argument(
        '--dataset',
        default='PTDB',
        help='The name of the dataset')
    parser.add_argument(
        '--rpa_threshold',
        type=float,
        default=0.84,
        help='Only keep examples with rpa score above this threshold')
    parser.add_argument(
        '--f1_threshold',
        type=float,
        default=0.88,
        help='Only keep examples with f1 score above this threshold')
    return parser.parse_args()

def main():
    args = parse_args()
    clean_data(args.results, args.partition, args.dataset, args.rpa_threshold, args.f1_threshold)

if __name__ == '__main__':
    main()
