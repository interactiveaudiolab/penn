import json
import time

import penne


###############################################################################
# Evaluate
###############################################################################


def datasets(datasets, checkpoint=penne.DEFAULT_CHECKPOINT, gpu=None):
    """Perform evaluation"""
    # Start benchmarking
    penne.BENCHMARK = True
    penne.TIMER.reset()
    start = time.time()

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

            # Infer
            _, _, logits = penne.from_audio(
                audio[0],
                penne.SAMPLE_RATE,
                model=penne.MODEL,
                checkpoint=checkpoint,
                batch_size=2048,
                gpu=gpu)

            logits = []

            # Preprocess audio
            iterator = penne.preprocess(
                audio,
                penne.SAMPLE_RATE,
                model=penne.MODEL,
                batch_size=2048)
            for frames in iterator:

                # Copy to device
                with penne.time.timer('copy-to'):
                    frames = frames.to('cpu' if gpu is None else f'cuda:{gpu}')

                # Infer
                logits.append(infer(frames, model, checkpoint).detach())

            # Concatenate results
            logits = torch.cat(logits, 2)

            # Update metrics
            file_metrics.update(logits, bins, pitch, voiced)
            dataset_metrics.update(logits, bins, pitch, voiced)
            aggregate_metrics.update(logits, bins, pitch, voiced)

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
    benchmark['elapsed'] = time.time() - start

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
