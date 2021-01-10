"""partition.py - dataset partitioning"""


import argparse
import json

import penne


###############################################################################
# Partition
###############################################################################


def dataset(name):
    """Partition a dataset

    Arguments
        name - string
            The name of the dataset

    Returns
        partitions - dict(string, list(string))
            The resulting partitions. The key is the partition name and the
            value is the list of stems belonging to that partition.
    """
    # Get a list of filenames without extension to be partitioned
    # TODO - replace with your datasets
    if name == 'DATASET':
        stems = DATASET_stems()
    else:
        raise ValueError(f'Dataset {name} is not implemented')

    # Partition files
    return from_stems(stems)


def dataset_to_file(name):
    """Partition dataset and write json file

    Arguments
        name - string
            The name of the dataset
    """
    # Create output directory
    output_directory = penne.ASSETS_DIR / name
    output_directory.mkdir(exist_ok=True, parents=True)

    # Write partition file
    with open(output_directory / 'partition.json', 'w') as file:
        json.dump(dataset(name), file)


def from_stems(stems):
    """Partition stems

    Arguments
        stems - list(string)
            The dataset file stems to partition

    Returns
        partitions - dict(string, list(string))
            The resulting partitions. The key is the partition name and the
            value is the list of stems belonging to that partition.
    """
    # TODO - partition the stems
    raise NotImplementedError


###############################################################################
# Dataset-specific
###############################################################################


def DATASET_stems():
    """Get a list of filenames without extension to be partitioned

    Returns
        stems - list(string)
            The list of file stems to partition
    """
    # TODO - return a list of stems for this dataset
    raise NotImplementedError


###############################################################################
# Entry point
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='The name of the dataset to partition')
    return parser.parse_args()


if __name__ == '__main__':
    dataset_to_file(parse_args().dataset)
