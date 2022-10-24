from torch.utils.tensorboard import SummaryWriter


###############################################################################
# Tensorboard logging
###############################################################################


def audio(directory, step, audio, sample_rate):
    """Write audio to Tensorboard"""
    for name, waveform in audio.items():
        writer(directory).add_audio(
            name,
            waveform,
            step,
            sample_rate)


def figures(directory, step, figures):
    """Write figures to Tensorboard"""
    for name, figure in figures.items():
        writer(directory).add_figure(name, figure, step)


def images(directory, step, images):
    """Write images to Tensorboard"""
    for name, image in images.items():
        writer(directory).add_image(name, image, step, dataformats='HWC')


def scalars(directory, step, scalars):
    """Write scalars to Tensorboard"""
    for name, scalar in scalars.items():
        writer(directory).add_scalar(name, scalar, step)


###############################################################################
# Utilities
###############################################################################


def writer(directory):
    """Get the writer object"""
    if not hasattr(writer, 'writer') or writer.directory != directory:
        writer.writer = SummaryWriter(log_dir=directory)
        writer.directory = directory
    return writer.writer
