import matplotlib.pyplot as plt
import torch

import penne


###############################################################################
# Plot dataset density vs true positive density
###############################################################################


def to_file(
    true_datasets,
    inference_datasets,
    output_file,
    checkpoint=None,
    gpu=None):
    """Plot ground truth and true positive densities"""
    # TODO - styling
    figure = plt.figure()

    # Plot true data density
    plt.hist(true_histogram(true_datasets, gpu), alpha=0.5)

    # Plot our correct guesses
    plt.hist(
        inference_histogram(inference_datasets, checkpoint, gpu),
        alpha=0.5)

    # Save plot
    figure.save(output_file, bbox_inches='tight', pad_inches=0)


def inference_histogram(datasets, checkpoint=None, gpu=None):
    """Get histogram of true positives from datasets and model checkpoint"""
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

    # Initialize counts
    counts = torch.zeros((penne.PITCH_BINS,))

    # Setup loader
    loader = penne.data.loader(datasets, 'test', gpu)

    # Update counts
    for audio, bins, _, voiced, _ in loader:

        # Preprocess audio
        batch_size = \
            None if gpu is None else penne.EVALUATION_BATCH_SIZE
        iterator = penne.preprocess(
            audio[0],
            penne.SAMPLE_RATE,
            model=penne.MODEL,
            batch_size=batch_size)
        for i, (frames, size) in enumerate(iterator):

            # Copy to device
            frames = frames.to(device)

            # Slice features and copy to GPU
            start = i * penne.EVALUATION_BATCH_SIZE
            end = start + size
            batch_bins = bins[:, start:end].to(device)
            batch_voiced = voiced[:, start:end].to(device)

            # Infer
            batch_logits = penne.infer(
                frames,
                penne.MODEL,
                checkpoint).detach()

            # Get predicted bins
            batch_predicted, *_ = penne.postprocess(batch_logits)

            # Get true positives
            true_positives = batch_predicted[
                batch_voiced & (batch_predicted == batch_bins)]

            # Update counts
            counts += torch.histogram(true_positives)

    return counts


def true_histogram(datasets, gpu=None):
    """Get histogram of ground truth from datasets"""
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

    # Initialize counts
    counts = torch.zeros((penne.PITCH_BINS,))

    # Setup loader
    loader = penne.data.loader(datasets, 'test', gpu)

    # Update counts
    for _, bins, _, _, _ in loader:
        counts += torch.histogram(bins.to(device), penne.PITCH_BINS)

    return counts
