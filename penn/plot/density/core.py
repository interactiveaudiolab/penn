import matplotlib.pyplot as plt
import torch

import penn


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
    plt.hist(true_histogram(true_datasets), alpha=0.5)

    # Plot our correct guesses
    plt.hist(
        inference_histogram(inference_datasets, checkpoint, gpu),
        alpha=0.5)

    # Save plot
    figure.savefig(output_file, bbox_inches='tight', pad_inches=0)


def inference_histogram(datasets, checkpoint=None, gpu=None):
    """Get histogram of true positives from datasets and model checkpoint"""
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

    # Initialize counts
    counts = torch.zeros((penn.PITCH_BINS,))

    # Setup loader
    loader = penn.data.loader(datasets, 'test', gpu)

    # Update counts
    for audio, bins, _, voiced, _ in loader:

        # Preprocess audio
        batch_size = \
            None if gpu is None else penn.EVALUATION_BATCH_SIZE
        iterator = penn.preprocess(
            audio[0],
            penn.SAMPLE_RATE,
            batch_size=batch_size)
        for i, (frames, size) in enumerate(iterator):

            # Copy to device
            frames = frames.to(device)

            # Slice features and copy to GPU
            start = i * penn.EVALUATION_BATCH_SIZE
            end = start + size
            batch_bins = bins[:, start:end].to(device)
            batch_voiced = voiced[:, start:end].to(device)

            # Infer
            batch_logits = penn.infer(frames, checkpoint).detach()

            # Get predicted bins
            batch_predicted, *_ = penn.postprocess(batch_logits)

            # Get true positives
            true_positives = batch_predicted[
                batch_voiced & (batch_predicted == batch_bins)]

            # Update counts
            counts += torch.histogram(
                true_positives.cpu().float(), penn.PITCH_BINS)[0]

    return counts


def true_histogram(datasets):
    """Get histogram of ground truth from datasets"""
    # Initialize counts
    counts = torch.zeros((penn.PITCH_BINS,))

    # Setup loader
    loader = penn.data.loader(datasets, 'test')

    # Update counts
    for _, bins, _, _, _ in loader:
        counts += torch.histogram(bins.float(), penn.PITCH_BINS)[0]

    return counts
