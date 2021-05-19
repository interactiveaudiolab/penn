"""evaluate.py - model evaluation"""


import argparse
import json
from pathlib import Path
import numpy as np
import torch
import tqdm

import penne


###############################################################################
# Evaluate
###############################################################################


def dataset(name, partition, model, device):
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
    # files = [penne.data.stem_to_file(name, stem) for stem in stems]

    # Partition files
    return from_stems(name, model, stems, device)


def dataset_to_file(name, partition, model, file, device):
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
        json.dump(dataset(name, partition, model, device), file)


def from_stems(name, model, stems, device):
    """Evaluate files

    Arguments
        model - penne.Model
            The model to evaluate
        stems - list(string)
            The stems to evaluate

    Returns
        results - dict
            The dictionary of results. The key is the name of a metric and
            the value is the value received for that metric. Must be JSON
            serializable.
    """
    # setup metrics
    thresh = penne.threshold.Hysteresis()
    f1 = penne.metrics.F1(thresh)
    wrmse = penne.metrics.WRMSE()
    rpa = penne.metrics.RPA()
    rca = penne.metrics.RCA()

    # loop over stems
    for stem in tqdm.tqdm(stems, dynamic_ncols=True, desc="Evaluating"):
        # get file paths
        audio_file = penne.data.stem_to_file(name, stem)
        annotation_file = penne.data.stem_to_annotation(name, stem)

        fmax = 550. if name == 'PTDB' else penne.MAX_FMAX
        
        # get model-predicted pitch
        pitch, periodicity = penne.predict_from_file(audio_file, model=model, batch_size=1024, return_periodicity=True, device=device)#, decoder=penne.decode.argmax, fmax=fmax)

        # get annotated pitch
        annotation = penne.load.pitch_annotation(name, annotation_file)

        np_pitch = pitch.numpy()
        np_periodicity = periodicity.numpy()
        np_annotation = annotation.numpy()

        # offset to empirical best alignment since PTDB annotations are not the expected length for 10ms hopsize
        if name == 'PTDB':
            np_pitch = np_pitch[:,1:1+np_annotation.shape[1]]
            np_periodicity = np_periodicity[:,1:1+np_annotation.shape[1]]

        # update metrics
        f1.update(np_pitch, np_annotation, np_periodicity)
        wrmse.update(np_pitch, np_annotation, np_periodicity)
        rpa.update(np_pitch, np_annotation)
        rca.update(np_pitch, np_annotation)

    # compute final metrics
    precision, recall, f1_val = f1()
    wrmse_val = wrmse()
    rpa_val = rpa()
    rca_val = rca()
    
    results = {'precision': precision, 'recall': recall, 'f1': f1_val, 'wrmse': wrmse_val, 'rpa': rpa_val, 'rca': rca_val}
    return results


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
    parser.add_argument(
        'device',
        help='The device to use for evaluation'
    )

    return parser.parse_args()


def main():
    """Evaluate a model"""
    # Parse command-line arguments
    args = parse_args()

    # Setup model
    model = penne.Model.load_from_checkpoint(args.checkpoint)

    # Evaluate
    dataset_to_file(args.dataset, args.partition, model, args.file, args.device)


if __name__ == '__main__':
    main()
