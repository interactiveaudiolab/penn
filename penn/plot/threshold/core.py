import matplotlib.pyplot as plt
import torch

import penn


###############################################################################
# Create figure
###############################################################################


def from_datasets(output_file, checkpoint=None, gpu=None):
    """Plot periodicity thresholds"""
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

    # Setup loader
    loader = penn.data.loader(penn.DATASETS, 'test', gpu)

    # Setup metric
    metrics = penn.evaluate.metrics.F1()

    for audio, _, _, voiced, _ in loader:

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
            batch_voiced = voiced[:, start:end].to(device)

            # Infer
            batch_logits = penn.infer(frames, checkpoint).detach()

            # Decode periodicity
            _, _, periodicity = penn.postprocess(batch_logits)

            # Update metrics
            metrics.update(periodicity, batch_voiced)

    # Get thresholds and corresponding F1 values
    x, y = zip(*
        [(key, val) for key, val in metrics().items() if key.startswith('f1')])

    # Plot
    # TODO - styling
    figure = plt.figure()
    plt.plot(x, y)

    # Save
    figure.savefig(output_file, bbox_inches='tight', pad_inches=0)
