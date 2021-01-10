"""evaluate.py - model evaluation"""


import argparse
import json
from pathlib import Path

import penne


###############################################################################
# Evaluate
###############################################################################


def dataset(name, partition, model):
    """Evaluate a dataset

    Arguments
        name - string
            The name of the dataset
        partition - string
            The partition to evaluate

    Returns
        results - dict
            The dictionary of results. The key is the name of a metric and
            the value is the value received for that metric. Must be JSON
            serializable.
    """
    # Get stems for this partition
    stems = penne.data.partitions(name)[partition]

    # Resolve stems to filenames
    files = [penne.data.stem_to_file(name, stem) for stem in stems]

    # Partition files
    return from_files(model, files)


def dataset_to_file(name, partition, model, file):
    """Evaluate dataset and write results to json file

    Arguments
        name - string
            The name of the dataset
        partition - string
            The partition to evaluate
        model - NAME.Model
            The model to evaluate
        file - Path
            The json file to save results to
    """
    with open(file, 'w') as file:
        json.dump(dataset(name, partition, model), file)


def from_files(model, files):
    """Evaluate files

    Arguments
        model - NAME.Model
            The model to evaluate
        files - list(string)
            The files to evaluate

    Returns
        results - dict
            The dictionary of results. The key is the name of a metric and
            the value is the value received for that metric. Must be JSON
            serializable.
    """
    # TODO - evaluate
    raise NotImplementedError


###############################################################################
# Entry point
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dataset',
        help='The name of the dataset to evaluate')
    parser.add_argument(
        'partition',
        help='The partition to evaluate',
        default='valid')
    parser.add_argument(
        'checkpoint',
        type=Path,
        help='The checkpoint file to evaluate')
    parser.add_argument(
        'file',
        type=Path,
        help='The file to write results to')

    return parser.parse_args()


def main():
    """Evaluate a model"""
    # Parse command-line arguments
    args = parse_args()

    # Setup model
    model = penne.Model.load_from_checkpoint(args.checkpoint)

    # Evaluate
    dataset_to_file(args.dataset, args.partition, model, args.file)


if __name__ == '__main__':
    main()
