import itertools

import numpy as np
import torchaudio

import penne


###############################################################################
# Constants
###############################################################################


# MDB analysis parameters
MDB_HOPSIZE = 128 / 44100  # seconds

# PTDB analysis parameters
PTDB_HOPSIZE = 160  # samples
PTDB_WINDOW_SIZE = 512  # samples


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
    audio_files = (penne.DATA_DIR / 'mdb'/ 'audio_stems').glob('*.wav')
    audio_files = sorted([
        file for file in audio_files if not file.stem.startswith('._')])

    # Get pitch files
    pitch_files = [
        file.parent.parent /
        'annotation_stems' /
        file.with_suffix('.csv').name
        for file in audio_files]

    # Create cache
    output_directory = penne.CACHE_DIR / 'mdb'
    output_directory.mkdir(exist_ok=True, parents=True)

    # Write audio and pitch to cache
    iterator = penne.iterator(
        enumerate(zip(audio_files, pitch_files)),
        'Preprocessing mdb',
        total=len(audio_files))
    for i, (audio_file, pitch_file) in iterator:
        stem = f'{i:06d}'

        # Load and resample audio
        audio = penne.load.audio(audio_file)

        # Save to cache
        np.save(
            output_directory / f'{stem}-audio.npy',
            audio.numpy().squeeze())

        # Save audio for listening and evaluation
        torchaudio.save(
            output_directory / f'{stem}.wav',
            audio,
            penne.SAMPLE_RATE)

        # Load pitch
        annotations = np.loadtxt(open(pitch_file), delimiter=',')
        times, pitch = annotations[:, 0], annotations[:, 1]

        # Fill unvoiced regions via linear interpolation
        pitch, voiced = interpolate_unvoiced(pitch)

        # Get target number of frames
        frames = penne.convert.samples_to_frames(audio.shape[-1])

        # Linearly interpolate to target number of frames
        new_times = penne.HOPSIZE_SECONDS * np.arange(0, frames)
        new_times += penne.HOPSIZE_SECONDS / 2.
        pitch = 2. ** np.interp(new_times, times, np.log2(pitch))

        # Linearly interpolate voiced/unvoiced tokens
        voiced = np.interp(new_times, times, voiced) > .5

        # Check shapes
        assert (
            penne.convert.samples_to_frames(audio.shape[-1]) ==
            pitch.shape[-1] ==
            voiced.shape[-1])

        # Save to cache
        np.save(output_directory / f'{stem}-pitch.npy', pitch)
        np.save(output_directory / f'{stem}-voiced.npy', voiced)


def ptdb():
    """Preprocessing ptdb dataset"""
    # Get audio files
    directory = penne.DATA_DIR / 'ptdb' / 'SPEECH DATA'
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
    output_directory = penne.CACHE_DIR / 'ptdb'
    output_directory.mkdir(exist_ok=True, parents=True)

    # Write audio and pitch to cache
    iterator = penne.iterator(
        enumerate(zip(audio_files, pitch_files)),
        'Preprocessing ptdb',
        total=len(audio_files))
    for i, (audio_file, pitch_file) in iterator:
        stem = f'{i:06d}'

        # Load and resample audio
        audio = penne.load.audio(audio_file)

        # Simluate the common padding error
        np.save(
            output_directory / f'{stem}-misalign.npy',
            audio.numpy().squeeze())

        # Fix padding error
        offset = PTDB_WINDOW_SIZE - PTDB_HOPSIZE // 2
        if (audio.shape[-1] - 2 * offset) % penne.HOPSIZE == 0:
            offset += PTDB_HOPSIZE // 2
        audio = audio[:, offset:-offset]

        # Save to cache
        np.save(
            output_directory / f'{stem}-audio.npy',
            audio.numpy().squeeze())

        # Save audio for listening and evaluation
        torchaudio.save(
            output_directory / f'{stem}.wav',
            audio,
            penne.SAMPLE_RATE)

        # Load pitch
        pitch = np.loadtxt(open(pitch_file), delimiter=' ')[:, 0]

        # Fill unvoiced regions via linear interpolation
        pitch, voiced = interpolate_unvoiced(pitch)

        # Check shapes
        assert (
            penne.convert.samples_to_frames(audio.shape[-1]) ==
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
