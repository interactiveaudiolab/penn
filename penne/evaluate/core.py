import json

import torch

import penne


###############################################################################
# Evaluate
###############################################################################


def datasets(datasets, checkpoint=penne.DEFAULT_CHECKPOINT, gpu=None):
    """Perform evaluation"""
    device = 'cpu' if gpu is None else f'cuda:{gpu}'

    # Containers for results
    overall, granular = {}

    # Per-file metrics
    file_metrics = penne.evaluate.Metrics()

    # Per-dataset metrics
    dataset_metrics = penne.evaluate.Metrics()

    # Aggregate metrics over all datasets
    aggregate_metrics = penne.evaluate.Metrics()

    # Turn off gradients
    with torch.no_grad():

        # Evaluate each dataset
        for dataset in datasets:

            # Reset dataset metrics
            dataset_metrics.reset()

            # Setup test dataset
            iterator = penne.data.loader(dataset, 'test')

            # Iterate over test set
            for audio, bins, pitch, voiced, stem in iterator:

                # Reset file metrics
                file_metrics.reset()

                _, _, logits = penne.from_audio(
                    audio,
                    penne.SAMPLE_RATE,
                    checkpoint=checkpoint,
                    gpu=gpu)

                # Update metrics
                file_metrics.update(logits, bins, pitch, voiced)
                dataset_metrics.update(logits, bins, pitch, voiced)
                aggregate_metrics.update(logits, bins, pitch, voiced)

                # Copy results
                granular[f'{dataset}/{stem}'] = file_metrics()

            # Copy results
            overall[dataset] = dataset_metrics()

    # Copy results
    overall['aggregate'] = aggregate_metrics()

    # Make output directory
    directory = penne.EVAL_DIR / penne.NAME
    directory.mkdir(exist_ok=True, parents=True)

    # Write to json files
    with open(directory / 'overall.json', 'w') as file:
        json.dump(overall, file, indent=4)
    with open(directory / 'granular.json', 'w') as file:
        json.dump(granular, file, indent=4)
