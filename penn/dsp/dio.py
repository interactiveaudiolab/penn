import functools
import multiprocessing as mp

import numpy as np
import torch

import penn


###############################################################################
# DIO (from pyworld)
###############################################################################


def from_audio(
    audio,
    sample_rate=penn.SAMPLE_RATE,
    hopsize=penn.HOPSIZE_SECONDS,
    fmin=penn.FMIN,
    fmax=penn.FMAX):
    """Estimate pitch and periodicity with dio"""
    with penn.time.timer('infer'):

        import pyworld

        # Convert to numpy
        audio = audio.numpy().squeeze().astype(np.float)

        # Get pitch
        pitch, times  = pyworld.dio(
            audio[penn.WINDOW_SIZE // 2:-penn.WINDOW_SIZE // 2],
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

        # Interpolate unvoiced tokens
        pitch, _ = penn.data.preprocess.interpolate_unvoiced(pitch)

        # Convert to torch
        return torch.from_numpy(pitch)[None]


def from_file(
        file,
        hopsize=penn.HOPSIZE_SECONDS,
        fmin=penn.FMIN,
        fmax=penn.FMAX):
    """Estimate pitch and periodicity with dio from audio on disk"""
    # Load
    with penn.time.timer('load'):
        audio = penn.load.audio(file)

    # Infer
    return from_audio(audio, penn.SAMPLE_RATE, hopsize, fmin, fmax)


def from_file_to_file(
        file,
        output_prefix=None,
        hopsize=penn.HOPSIZE_SECONDS,
        fmin=penn.FMIN,
        fmax=penn.FMAX):
    """Estimate pitch and periodicity with dio and save to disk"""
    # Infer
    results = from_file(file, hopsize, fmin, fmax)

    # Save to disk
    with penn.time.timer('save'):

        # Maybe use same filename with new extension
        if output_prefix is None:
            output_prefix = file.parent / file.stem

        # Save pitch
        torch.save(results[0], f'{output_prefix}-pitch.pt')

        # Maybe save periodicity
        if len(results) > 1:
            torch.save(results[1], f'{output_prefix}-periodicity.pt')


def from_files_to_files(
        files,
        output_prefixes=None,
        hopsize=penn.HOPSIZE_SECONDS,
        fmin=penn.FMIN,
        fmax=penn.FMAX):
    """Estimate pitch and periodicity with dio and save to disk"""
    pitch_fn = functools.partial(
        from_file_to_file,
        hopsize=hopsize,
        fmin=fmin,
        fmax=fmax)
    iterator = zip(files, output_prefixes)

    # Turn off multiprocessing for benchmarking
    if penn.BENCHMARK:
        iterator = penn.iterator(
            iterator,
            f'{penn.CONFIG}',
            total=len(files))
        for item in iterator:
            pitch_fn(*item)
    else:
        with mp.get_context('spawn').Pool() as pool:
            pool.starmap(pitch_fn, iterator)
