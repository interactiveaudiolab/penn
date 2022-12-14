import matplotlib.pyplot as plt
import torch

import penn


###############################################################################
# Constants
###############################################################################


# Amount of bin downsampling
DOWNSAMPLE_RATE = penn.PITCH_BINS // 90


###############################################################################
# Plot dataset density vs true positive density
###############################################################################


def to_file(
    datasets,
    output_file,
    checkpoint=None,
    gpu=None):
    """Plot ground truth and true positive densities"""
    figure, axis = plt.subplots()
    axis.set_axis_off()

    # Plot true data density
    x = torch.arange(0, penn.PITCH_BINS, DOWNSAMPLE_RATE)
    y_true, y_pred = histograms(datasets, checkpoint, gpu)
    y_true = y_true.reshape(-1, DOWNSAMPLE_RATE).sum(-1)
    axis.bar(
        x,
        y_true,
        alpha=0.5,
        width=DOWNSAMPLE_RATE,
        label=f'Data distribution')

    # Plot our correct guesses
    y_pred = y_pred.reshape(-1, DOWNSAMPLE_RATE).sum(-1)
    axis.bar(
        x,
        y_pred,
        alpha=0.5,
        width=DOWNSAMPLE_RATE,
        label='Inferred distribution')

    # Remove axes
    # axis.legend()

    # Save plot
    figure.savefig(output_file, bbox_inches='tight', pad_inches=0, dpi=300)


def histograms(datasets, checkpoint=None, gpu=None):
    """Get histogram of true positives from datasets and model checkpoint"""
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

    # Initialize counts
    true_result = torch.zeros((penn.PITCH_BINS,))
    infer_result = torch.zeros((penn.PITCH_BINS,))

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
            difference = torch.abs(
                penn.convert.bins_to_cents(batch_predicted) -
                penn.convert.bins_to_cents(batch_bins))
            # import pdb; pdb.set_trace()
            true_all = batch_bins[batch_voiced]
            true_positives = batch_bins[
                batch_voiced & (difference < penn.evaluate.metrics.THRESHOLD)]
            # true_positives = batch_bins[batch_voiced & (batch_predicted == batch_bins)]

            # Update counts
            true_result += torch.histogram(
                true_all.cpu().float(), penn.PITCH_BINS)[0]
            infer_result += torch.histogram(
                true_positives.cpu().float(), penn.PITCH_BINS)[0]

    return true_result, infer_result
