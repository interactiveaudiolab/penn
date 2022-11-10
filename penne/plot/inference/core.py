import matplotlib.pyplot as plt

import penne


###############################################################################
# Create figure
###############################################################################


def from_audio(audio, sample_rate, gpu=None):
    """Plot pitch and periodicity"""
    # Get pitch and periodicity
    pitch, periodicity = penne.from_audio(audio, sample_rate, gpu=gpu)

    # Setup figure
    figure, axes = plt.subplots(3, 1, figsize=(18, 6))

    # Plot waveform
    axes[0].plot(audio.squeeze(), color='black', linewidth=.5)
    axes[0].set_axis_off()
    axes[0].set_ylim([-1., 1.])

    # Plot pitch
    axes[1].plot(pitch.squeeze(), color='black', linewidth=.5)
    axes[1].set_axis_off()

    # Plot periodicity
    axes[2].plot(periodicity.squeeze(), color='black', linewidth=.5)
    axes[2].set_axis_off()

    return figure


def from_file(audio_file, gpu=None):
    """Plot pitch and periodicity from file on disk"""
    # Load audio
    audio = penne.load.audio(audio_file)

    # Plot
    return from_audio(audio, penne.SAMPLE_RATE, gpu)


def from_file_to_file(audio_file, output_file, gpu=None):
    """Plot pitch and periodicity and save to disk"""
    # Plot
    figure = from_file(audio_file, gpu)

    # Save to disk
    figure.save(output_file, bbox_inches='tight', pad_inches=0)
