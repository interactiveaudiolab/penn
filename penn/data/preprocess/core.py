import itertools
import warnings

import numpy as np
import torchaudio

import penn


###############################################################################
# Constants
###############################################################################


# MDB analysis parameters
MDB_HOPSIZE = 128  # samples
MDB_SAMPLE_RATE = 44100  # samples per second

# PTDB analysis parameters
PTDB_HOPSIZE = 160  # samples
PTDB_SAMPLE_RATE = 16000  # samples per second
PTDB_WINDOW_SIZE = 512  # samples
PTDB_HOPSIZE_SECONDS = PTDB_HOPSIZE / PTDB_SAMPLE_RATE


###############################################################################
# Preprocess datasets
###############################################################################


def datasets(datasets):
    """Preprocess datasets"""
    if 'mdb' in datasets:
        mdb()

    if 'ptdb' in datasets:
        ptdb()


###############################################################################
# Individual datasets
###############################################################################


def mdb():
    """Preprocess mdb dataset"""
    # Get audio files
    audio_files = (penn.DATA_DIR / 'mdb'/ 'audio_stems').glob('*.wav')
    audio_files = sorted([
        file for file in audio_files if not file.stem.startswith('._')])

    # Get pitch files
    pitch_files = [
        file.parent.parent /
        'annotation_stems' /
        file.with_suffix('.csv').name
        for file in audio_files]

    # Create cache
    output_directory = penn.CACHE_DIR / 'mdb'
    output_directory.mkdir(exist_ok=True, parents=True)

    # Write audio and pitch to cache
    iterator = penn.iterator(
        enumerate(zip(audio_files, pitch_files)),
        'Preprocessing mdb',
        total=len(audio_files))
    for i, (audio_file, pitch_file) in iterator:
        stem = f'{i:06d}'

        # Load and resample audio
        audio = penn.load.audio(audio_file)

        # Save as numpy array for fast memory-mapped reads
        np.save(
            output_directory / f'{stem}-audio.npy',
            audio.numpy().squeeze())

        # Save audio for listening and evaluation
        torchaudio.save(
            output_directory / f'{stem}.wav',
            audio,
            penn.SAMPLE_RATE)

        # Load pitch
        annotations = np.loadtxt(open(pitch_file), delimiter=',')
        times, pitch = annotations[:, 0], annotations[:, 1]

        # Fill unvoiced regions via linear interpolation
        pitch, voiced = interpolate_unvoiced(pitch)

        # Get target number of frames
        frames = penn.convert.samples_to_frames(audio.shape[-1])

        # Linearly interpolate to target number of frames
        new_times = penn.HOPSIZE_SECONDS * np.arange(0, frames)
        new_times += penn.HOPSIZE_SECONDS / 2.
        pitch = 2. ** np.interp(new_times, times, np.log2(pitch))

        # Linearly interpolate voiced/unvoiced tokens
        voiced = np.interp(new_times, times, voiced) > .5

        # Check shapes
        assert (
            penn.convert.samples_to_frames(audio.shape[-1]) ==
            pitch.shape[-1] ==
            voiced.shape[-1])

        # Save to cache
        np.save(output_directory / f'{stem}-pitch.npy', pitch)
        np.save(output_directory / f'{stem}-voiced.npy', voiced)


def ptdb():
    """Preprocessing ptdb dataset"""
    # Get audio files
    directory = penn.DATA_DIR / 'ptdb' / 'SPEECH DATA'
    male = (directory / 'MALE' / 'MIC').rglob('*.wav')
    female = (directory / 'FEMALE' / 'MIC').rglob('*.wav')
    audio_files = sorted(itertools.chain(male, female))

    # Get pitch files
    pitch_files = [
        file.parent.parent.parent /
        'REF' /
        file.parent.name /
        file.with_suffix('.f0').name.replace('mic', 'ref')
        for file in audio_files]

    # Create cache
    output_directory = penn.CACHE_DIR / 'ptdb'
    output_directory.mkdir(exist_ok=True, parents=True)

    # Write audio and pitch to cache
    iterator = penn.iterator(
        enumerate(zip(audio_files, pitch_files)),
        'Preprocessing ptdb',
        total=len(audio_files))
    for i, (audio_file, pitch_file) in iterator:
        stem = f'{i:06d}'

        # Load and resample to PTDB sample rate
        audio, sample_rate = torchaudio.load(audio_file)
        audio = penn.resample(audio, sample_rate, PTDB_SAMPLE_RATE)

        # Remove padding
        offset = PTDB_WINDOW_SIZE - PTDB_HOPSIZE // 2
        if (audio.shape[-1] - 2 * offset) % PTDB_HOPSIZE == 0:
            offset += PTDB_HOPSIZE // 2
        audio = audio[:, offset:-offset]

        # Resample to pitch estimation sample rate
        audio = penn.resample(audio, PTDB_SAMPLE_RATE)

        # Save as numpy array for fast memory-mapped read
        np.save(
            output_directory / f'{stem}-audio.npy',
            audio.numpy().squeeze())

        # Save audio for listening and evaluation
        torchaudio.save(
            output_directory / f'{stem}.wav',
            audio,
            penn.SAMPLE_RATE)

        # Load pitch
        pitch = np.loadtxt(open(pitch_file), delimiter=' ')[:, 0]

        # Fill unvoiced regions via linear interpolation
        pitch, voiced = interpolate_unvoiced(pitch)

        # Get target number of frames
        frames = penn.convert.samples_to_frames(audio.shape[-1])

        # Get original times
        times = PTDB_HOPSIZE_SECONDS * np.arange(0, len(pitch))
        times += PTDB_HOPSIZE_SECONDS / 2

        # Linearly interpolate to target number of frames
        new_times = penn.HOPSIZE_SECONDS * np.arange(0, frames)
        new_times += penn.HOPSIZE_SECONDS / 2.

        pitch = 2. ** np.interp(new_times, times, np.log2(pitch))

        # Linearly interpolate voiced/unvoiced tokens
        voiced = np.interp(new_times, times, voiced) > .5

        # Check shapes
        assert (
            penn.convert.samples_to_frames(audio.shape[-1]) ==
            pitch.shape[-1] ==
            voiced.shape[-1])

        # Save to cache
        np.save(output_directory / f'{stem}-pitch.npy', pitch)
        np.save(output_directory / f'{stem}-voiced.npy', voiced)


###############################################################################
# Utilities
###############################################################################


def interpolate_unvoiced(pitch):
    """Fill unvoiced regions via linear interpolation"""
    unvoiced = pitch == 0

    # Ignore warning of log setting unvoiced regions (zeros) to nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        # Pitch is linear in base-2 log-space
        pitch = np.log2(pitch)

    try:

        # Interpolate
        pitch[unvoiced] = np.interp(
            np.where(unvoiced)[0],
            np.where(~unvoiced)[0],
            pitch[~unvoiced])

    except ValueError:

        # Allow all unvoiced
        pass

    return 2 ** pitch, ~unvoiced
