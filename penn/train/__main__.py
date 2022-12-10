import argparse
import shutil
from pathlib import Path

import penn


###############################################################################
# Entry point
###############################################################################


def main(config, datasets, gpus=None):
    # Create output directory
    directory = penn.RUNS_DIR / config.stem
    directory.mkdir(parents=True, exist_ok=True)

    # Save configuration
    shutil.copyfile(config, directory / config.name)

    # Train
    checkpoint = penn.train.run(
        datasets,
        directory,
        directory,
        directory,
        gpus)

    # Evaluate
    penn.evaluate.datasets(checkpoint=checkpoint, gpu=gpus[0])


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='The configuration file')
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=penn.DATASETS,
        help='The datasets to train on')
    parser.add_argument(
        '--gpus',
        type=int,
        nargs='+',
        help='The gpus to run training on')
    return parser.parse_args()


main(**vars(parse_args()))
