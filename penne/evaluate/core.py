import json
import time

import torch

import penne


###############################################################################
# Evaluate
###############################################################################


def datasets(datasets, checkpoint=penne.DEFAULT_CHECKPOINT, gpu=None):
    """Perform evaluation"""
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

    # Start benchmarking
    penne.BENCHMARK = True
    penne.TIMER.reset()
    start_time = time.time()

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

            # Preprocess audio
            iterator = penne.preprocess(
                audio[0],
                penne.SAMPLE_RATE,
                model=penne.MODEL,
                batch_size=penne.EVALUATION_BATCH_SIZE)
            for i, frames in enumerate(iterator):

                # Copy to device
                with penne.time.timer('copy-to'):
                    frames = frames.to(device)

                # Slice features and copy to GPU
                with penne.time.timer('copy-eval'):
                    start = i * penne.EVALUATION_BATCH_SIZE
                    end = start + frames.shape[0]
                    batch_bins = bins[:, start:end].to(device)
                    batch_pitch = pitch[:, start:end].to(device)
                    batch_voiced = voiced[:, start:end].to(device)

                # Infer
                logits = penne.infer(
                    frames,
                    penne.MODEL,
                    checkpoint).detach()

                # Update metrics
                with penne.time.timer('eval'):
                    args = logits, batch_bins, batch_pitch, batch_voiced
                    file_metrics.update(*args)
                    dataset_metrics.update(*args)
                    aggregate_metrics.update(*args)

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

    # Turn off benchmarking
    penne.BENCHMARK = False

    # Get benchmarking information
    benchmark = penne.TIMER()
    benchmark['elapsed'] = time.time() - start_time

    # Get total number of frames, samples, and seconds in test data
    frames = aggregate_metrics.loss.count
    samples = penne.convert.frames_to_samples(frames)
    seconds = penne.convert.frames_to_seconds(frames)

    # Format benchmarking results
    results = {
        key: {
            'real-time-factor': seconds / value,
            'samples': samples,
            'samples-per-second': samples / value,
            'total': value
        } for key, value in benchmark.items()}

    # Write benchmarking information
    with open(directory / 'time.json', 'w') as file:
        json.dump(results, file, indent=4)
