import functools
import multiprocessing as mp

import torch

import penne


###############################################################################
# DIO (from pyworld)
###############################################################################


def from_audio(
    audio,
    sample_rate=penne.SAMPLE_RATE,
    hopsize=penne.HOPSIZE_SECONDS,
    fmin=penne.FMIN,
    fmax=penne.FMAX):
    """Estimate pitch and periodicity with dio"""
    import pyworld

    # Convert to numpy
    audio = audio.numpy()

    # Get pitch
    pitch, times  = pyworld.dio(
        audio,
        sample_rate,
        fmin,
        fmax,
        frame_period=1000 * hopsize)

    # Refine pitch
    pitch = pyworld.stonemask(
        audio,
        pitch,
        times,
        sample_rate)

    # Convert to torch
    return torch.from_numpy(pitch)


def from_file(
        file,
        hopsize=penne.HOPSIZE_SECONDS,
        fmin=penne.FMIN,
        fmax=penne.FMAX):
    """Estimate pitch and periodicity with dio from audio on disk"""
    # Load
    with penne.time.timer('load'):
        audio = penne.load.audio(file)

    # Infer
    return from_audio(audio, penne.SAMPLE_RATE, hopsize, fmin, fmax)


def from_file_to_file(
        file,
        output_prefix=None,
        hopsize=penne.HOPSIZE_SECONDS,
        fmin=penne.FMIN,
        fmax=penne.FMAX):
    """Estimate pitch and periodicity with dio and save to disk"""
    # Infer
    pitch, periodicity = from_file(file, hopsize, fmin, fmax)

    # Save to disk
    with penne.time.timer('save'):

        # Maybe use same filename with new extension
        if output_prefix is None:
            output_prefix = file.parent / file.stem

        # Save
        torch.save(pitch, f'{output_prefix}-pitch.pt')
        torch.save(periodicity, f'{output_prefix}-periodicity.pt')


def from_files_to_files(
        files,
        output_prefixes=None,
        hopsize=penne.HOPSIZE_SECONDS,
        fmin=penne.FMIN,
        fmax=penne.FMAX):
    """Estimate pitch and periodicity with dio and save to disk"""
    pitch_fn = functools.partial(
        from_file_to_file,
        hopsize=hopsize,
        fmin=fmin,
        fmax=fmax)
    iterator = zip(files, output_prefixes)

    # Turn off multiprocessing for benchmarking
    if penne.BENCHMARK:
        iterator = penne.iterator(
            iterator,
            f'{penne.CONFIG}',
            total=len(files))
        for item in iterator:
            pitch_fn(*item)
    else:
        with mp.get_context('spawn').Pool() as pool:
            pool.starmap(pitch_fn, iterator)
