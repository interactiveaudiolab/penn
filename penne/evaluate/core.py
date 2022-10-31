import json
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

import penne


###############################################################################
# Evaluate
###############################################################################


def datasets(
    datasets,
    method='cd-crepe',
    checkpoint=penne.DEFAULT_CHECKPOINT,
    gpu=None):
    """Perform evaluation"""
    quality(datasets, method, checkpoint, gpu)

    # Benchmarking after the quality evaluation ensure model loading time is
    # not included in benchmarking
    benchmark(datasets, method, checkpoint, gpu)


###############################################################################
# Utilities
###############################################################################


def benchmark(
    datasets,
    method='cd-crepe',
    checkpoint=penne.DEFAULT_CHECKPOINT,
    gpu=None):
    """Perform benchmarking"""
    # Get audio files
    dataset_stems = {
        dataset: penne.load.partition(dataset)['test'] for dataset in datasets}
    files = [
        penne.CACHE_DIR / dataset / f'{stem}.wav'
        for dataset, stems in dataset_stems.items()
        for stem in stems]

    # Setup temporary directory
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)

        # Create output directories
        for dataset in datasets:
            (directory / dataset).mkdir(exist_ok=True, parents=True)

        # Get output prefixes
        output_prefixes = [
            directory / file.parent.name / file.stem for file in files]

        # Start benchmarking
        start_time = time.time()

        # Infer to temporary storage
        if method == 'penne':
            penne.from_files_to_files(
                files,
                output_prefixes,
                checkpoint=checkpoint,
                batch_size=penne.EVALUATION_BATCH_SIZE,
                gpu=gpu)

        # TEMPORARY - include torchcrepe depdenecy
        elif method == 'torchcrepe':

            # Get output file paths
            pitch_files = [file.parent / f'{file.stem}-pitch.pt' for file in files]
            periodicity_files = [
                file.parent / f'{file.stem}-periodicity.pt' for file in files]

            # Infer
            # Note - this does not correctly handle padding, but suffices for
            #        benchmarking purposes
            torchcrepe.from_files_to_files(
                files,
                pitch_files,
                output_periodicity_files=periodicity_files,
                hop_length=penne.HOPSIZE,
                decoder=torchcrepe.decoder.argmax,
                batch_size=penne.EVALUATION_BATCH_SIZE,
                device='cpu' if gpu is None else f'cuda:{gpu}')

        elif method == 'harmof0':
            penne.temp.harmof0.from_files_to_files(
                # TODO
            )
        elif method == 'pyin':
            penne.dsp.pyin.from_files_to_files(
                # TODO
            )

        # Stop benchmarking
        elapsed = time.time() - start_time

    # Make output directory
    directory = penne.EVAL_DIR / penne.CONFIG
    directory.mkdir(exist_ok=True, parents=True)

    # Get total number of samples and seconds in test data
    samples = sum([
        len(np.load(file.parent / f'{file.stem}-audio.npy', mmap_mode='r'))
        for file in files])
    seconds = penne.convert.samples_to_seconds(samples)

    # Format benchmarking results
    results = {
        'real-time-factor': seconds / elapsed,
        'samples': samples,
        'samples-per-second': samples / elapsed,
        'total': elapsed}

    # Write benchmarking information
    with open(directory / 'time.json', 'w') as file:
        json.dump(results, file, indent=4)


def quality(
    datasets,
    method='cd-crepe',
    checkpoint=penne.DEFAULT_CHECKPOINT,
    gpu=None):
    """Evaluate pitch and periodicity estimation quality"""
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

    # Containers for results
    overall, granular = {}, {}

    # Per-file metrics
    file_metrics = penne.evaluate.Metrics()

    # Per-dataset metrics
    dataset_metrics = penne.evaluate.Metrics()

    # Aggregate metrics over all datasets
    aggregate_metrics = penne.evaluate.Metrics()

    # Evaluate each dataset
    for dataset in datasets:

        # Reset dataset metrics
        dataset_metrics.reset()

        # Setup test dataset
        iterator = penne.iterator(
            penne.data.loader([dataset], 'test'),
            f'Evaluating {penne.CONFIG} on {dataset}')

        # Iterate over test set
        for audio, bins, pitch, voiced, stem in iterator:

            # Reset file metrics
            file_metrics.reset()

            if method == 'cd-crepe':

                # Preprocess audio
                iterator = penne.preprocess(
                    audio[0],
                    penne.SAMPLE_RATE,
                    model=penne.MODEL,
                    batch_size=penne.EVALUATION_BATCH_SIZE)
                for i, (frames, size) in enumerate(iterator):

                    # Copy to device
                    frames = frames.to(device)

                    # Slice features and copy to GPU
                    start = i * penne.EVALUATION_BATCH_SIZE
                    end = start + size
                    batch_bins = bins[:, start:end].to(device)
                    batch_pitch = pitch[:, start:end].to(device)
                    batch_voiced = voiced[:, start:end].to(device)

                    # Infer
                    logits = penne.infer(
                        frames,
                        penne.MODEL,
                        checkpoint).detach()

                    # Update metrics
                    args = logits, batch_bins, batch_pitch, batch_voiced
                    file_metrics.update(*args)
                    dataset_metrics.update(*args)
                    aggregate_metrics.update(*args)

            elif method == 'torchcrepe':

                # TODO
                pass

            elif method == 'harmof0':

                # TODO
                pass

            elif method == 'pyin':

                # TODO
                pass

            # Copy results
            granular[f'{dataset}/{stem[0]}'] = file_metrics()
        overall[dataset] = dataset_metrics()
    overall['aggregate'] = aggregate_metrics()

    # Make output directory
    directory = penne.EVAL_DIR / penne.CONFIG
    directory.mkdir(exist_ok=True, parents=True)

    # Write to json files
    with open(directory / 'overall.json', 'w') as file:
        json.dump(overall, file, indent=4)
    with open(directory / 'granular.json', 'w') as file:
        json.dump(granular, file, indent=4)
