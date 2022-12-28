import torch

import penn


###############################################################################
# Create figure
###############################################################################


def from_audio(
    audio,
    sample_rate,
    checkpoint=penn.DEFAULT_CHECKPOINT,
    gpu=None):
    """Plot logits with pitch overlay"""
    import matplotlib
    import matplotlib.pyplot as plt

    logits = []

    # Change font size
    matplotlib.rcParams.update({'font.size': 16})

    # Preprocess audio
    iterator = penn.preprocess(
        audio,
        sample_rate,
        batch_size=penn.EVALUATION_BATCH_SIZE,
        pad=True)
    for frames, _ in iterator:

        # Copy to device
        frames = frames.to('cpu' if gpu is None else f'cuda:{gpu}')

        # Infer
        logits.append(penn.infer(frames, checkpoint=checkpoint).detach())

    # Concatenate results
    logits = torch.cat(logits)

    # Convert to distribution
    # NOTE - We use softmax even if the loss is BCE for more comparable
    #        visualization. Otherwise, the variance of models trained with
    #        BCE looks erroneously lower.
    distributions = torch.nn.functional.softmax(logits, dim=1)

    # Take the log again for display
    distributions = torch.log(distributions)
    distributions[torch.isinf(distributions)] = \
        distributions[~torch.isinf(distributions)].min()

    # Prepare for plotting
    distributions = distributions.cpu().squeeze(2).T

    # Setup figure
    figure, axis = plt.subplots(figsize=(18, 2))

    # Make pretty
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.spines['bottom'].set_visible(False)
    axis.spines['left'].set_visible(False)
    xticks = torch.arange(0, len(logits), int(penn.SAMPLE_RATE / penn.HOPSIZE))
    xlabels = xticks // 100
    axis.get_xaxis().set_ticks(xticks.tolist(), xlabels.tolist())
    yticks = torch.linspace(0, penn.PITCH_BINS - 1, 5)
    ylabels = penn.convert.bins_to_frequency(yticks)
    ylabels = ylabels.round().int().tolist()
    axis.get_yaxis().set_ticks(yticks, ylabels)
    axis.set_xlabel('Time (seconds)')
    axis.set_ylabel('Frequency (Hz)')

    # Plot pitch posteriorgram
    axis.imshow(distributions, aspect='auto', origin='lower')

    return figure


def from_file(audio_file, checkpoint=penn.DEFAULT_CHECKPOINT, gpu=None):
    """Plot logits and optional pitch"""
    # Load audio
    audio = penn.load.audio(audio_file)

    # Plot
    return from_audio(audio, penn.SAMPLE_RATE, checkpoint, gpu)


def from_file_to_file(
    audio_file,
    output_file,
    checkpoint=penn.DEFAULT_CHECKPOINT,
    gpu=None):
    """Plot pitch and periodicity and save to disk"""
    # Plot
    figure = from_file(audio_file, checkpoint, gpu)

    # Save to disk
    figure.savefig(output_file, bbox_inches='tight', pad_inches=0, dpi=900)
