import matplotlib
import matplotlib.pyplot as plt
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
    logits = []

    # Preprocess audio
    iterator = penn.preprocess(
        audio,
        sample_rate,
        batch_size=penn.EVALUATION_BATCH_SIZE)
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

    # TEMPORARY - take the log again for display?
    distributions = torch.log(distributions)
    distributions[torch.isinf(distributions)] = \
        distributions[~torch.isinf(distributions)].min()

    # Prepare for plotting
    distributions = distributions.cpu().squeeze(2).T.flipud()


    # Setup figure
    figure, axis = plt.subplots(figsize=(18, 2))

    # Make pretty
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.spines['bottom'].set_visible(False)
    axis.spines['left'].set_visible(False)
    # duration = audio.shape[-1] / sample_rate
    # step = max(1, duration // 3)
    # xticks = torch.arange(0, duration, step)
    # axis.get_xaxis().set_ticks(
    #     penn.convert.seconds_to_frames(xticks).int().tolist(),
    #     xticks.int().tolist())
    axis.get_xaxis().set_ticks([])
    yticks = torch.tensor(
        list(range(0, penn.PITCH_BINS, penn.PITCH_BINS // 4)) +
        [penn.PITCH_BINS]) - 1
    ylabels = penn.convert.bins_to_frequency(penn.PITCH_BINS - 1 - yticks)
    ylabels = ylabels.round().int().tolist()
    axis.get_yaxis().set_ticks(yticks, ylabels)
    # axis.set_ylabel('Frequency (Hz)')

    # Plot pitch posteriorgram
    axis.imshow(distributions, aspect='auto')

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
