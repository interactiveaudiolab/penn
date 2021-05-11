"""partition.py - dataset partitioning"""


import argparse
import json
import random
import math
import os
from pathlib import Path

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
    if name == 'MDB':
        stems = MDB_stems()
    elif name == 'PTDB':
        stems = PTDB_stems()
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
    # define partition percentages
    percents = {"train": .70, "valid": .15, "test": .15}
    partitions = {}
    
    # randomly shuffle all stems
    random.seed(0)
    random.shuffle(stems)
    num_stems = len(stems)

    # partition shuffled stems based on percentages
    num_valid = math.floor(percents["valid"] * num_stems)
    num_test = math.floor(percents["test"] * num_stems)
    partitions["valid"] = stems[0:num_valid]
    partitions["test"] = stems[num_valid:num_valid + num_test]
    partitions["train"] = stems[num_valid + num_test:]

    return partitions


###############################################################################
# Dataset-specific
###############################################################################

def MDB_stems():
    # look through MDB data directory and generate list of stems
    audio_dir = os.path.join(penne.DATA_DIR, "MDB", "audio_stems")
    stems = []
    for name in os.listdir(audio_dir):
        if name[0] != '.':
            stems.append(name[:name.index(".RESYN.wav")])
    return stems

def PTDB_stems():
    # look through PTDB data directory and generate list of stems
    stems = []
    for root, dirs, files in os.walk(os.path.join(penne.DATA_DIR, "PTDB")):
        if "LAR" in root:
            curr_stems = [name[name.index("_")+1:name.index(".wav")] for name in files]
            stems += curr_stems
    return stems


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
