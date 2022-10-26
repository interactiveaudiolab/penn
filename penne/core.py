from pathlib import Path
from typing import Optional, Tuple

import torch
import torchaudio
import tqdm

import penne


###############################################################################
# Crepe pitch prediction
###############################################################################


def from_audio(
    audio: torch.Tensor,
    sample_rate: int,
    hopsize: float = penne.HOPSIZE / penne.SAMPLE_RATE,
    fmin: float = penne.FMIN,
    fmax: Optional[float] = None,
    model: str = 'crepe',
    checkpoint: Path = penne.DEFAULT_CHECKPOINT,
    batch_size: Optional[int] = None,
    device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    """Perform pitch and periodicity estimation

    Args:
        audio: The audio to extract pitch and periodicity from
        sample_rate: The audio sample rate
        hopsize: The hopsize in seconds
        fmin: The minimum allowable frequency in Hz
        fmax: The maximum allowable frequency in Hz
        model: The name of the model
        checkpoint: The checkpoint file
        batch_size: The number of frames per batch
        device: The device used to run inference

    Returns:
        pitch: torch.tensor(shape=(1, int(samples // hopsize)))
        periodicity: torch.tensor(shape=(1, int(samples // hopsize)))
    """
    # TODO
    pass


def from_file(
    file: Path,
    hopsize: float = penne.HOPSIZE / penne.SAMPLE_RATE,
    fmin: float = penne.FMIN,
    fmax: Optional[float] = None,
    model: str = 'crepe',
    checkpoint: Path = penne.DEFAULT_CHECKPOINT,
    batch_size: Optional[int] = None,
    device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    """Perform pitch and periodicity estimation from audio on disk

    Args:
        file: The audio file
        hopsize: The hopsize in seconds
        fmin: The minimum allowable frequency in Hz
        fmax: The maximum allowable frequency in Hz
        model: The name of the model
        checkpoint: The checkpoint file
        batch_size: The number of frames per batch
        device: The device used to run inference

    Returns:
        pitch: torch.tensor(shape=(1, int(samples // hopsize)))
        periodicity: torch.tensor(shape=(1, int(samples // hopsize)))
    """
    # Load audio
    audio = penne.load.audio(file)

    # Inference
    return from_audio(
        audio,
        penne.SAMPLE_RATE,
        hopsize,
        fmin,
        fmax,
        model,
        checkpoint,
        batch_size,
        device)


def from_file_to_file(
    file: Path,
    output_prefix: Optional[Path] = None,
    hopsize: float = penne.HOPSIZE / penne.SAMPLE_RATE,
    fmin: float = penne.FMIN,
    fmax: Optional[float] = None,
    model: str = 'crepe',
    checkpoint: Path = penne.DEFAULT_CHECKPOINT,
    batch_size: Optional[int] = None,
    device: str = 'cpu') -> None:
    """Perform pitch and periodicity estimation from audio on disk and save

    Args:
        file: The audio file
        output_prefix: The file to save pitch and periodicity without extension
        hopsize: The hopsize in seconds
        fmin: The minimum allowable frequency in Hz
        fmax: The maximum allowable frequency in Hz
        model: The name of the model
        checkpoint: The checkpoint file
        batch_size: The number of frames per batch
        device: The device used to run inference

    Returns:
        pitch: torch.tensor(shape=(1, int(samples // hopsize)))
        periodicity: torch.tensor(shape=(1, int(samples // hopsize)))
    """
    # Inference
    pitch, periodicity = from_file(
        file,
        hopsize,
        fmin,
        fmax,
        model,
        checkpoint,
        batch_size,
        device)

    # Maybe use same filename with new extension
    if output_prefix is None:
        output_prefix = file.parent / file.stem

    # Save to disk
    torch.save(pitch, f'{output_prefix}-pitch.pt')
    torch.save(periodicity, f'{output_prefix}-periodicity.pt')


def from_files_to_files(
    files: Path,
    output_prefixes: Optional[list] = None,
    hopsize: float = penne.HOPSIZE / penne.SAMPLE_RATE,
    fmin: float = penne.FMIN,
    fmax: Optional[float] = None,
    model: str = 'crepe',
    checkpoint: Path = penne.DEFAULT_CHECKPOINT,
    batch_size: Optional[int] = None,
    device: str = 'cpu') -> None:
    """Perform pitch and periodicity estimation from files on disk and save

    Args:
        files: The audio files
        output_prefixes: Files to save pitch and periodicity without extension
        hopsize: The hopsize in seconds
        fmin: The minimum allowable frequency in Hz
        fmax: The maximum allowable frequency in Hz
        model: The name of the model
        checkpoint: The checkpoint file
        batch_size: The number of frames per batch
        device: The device used to run inference

    Returns:
        pitch: torch.tensor(shape=(1, int(samples // hopsize)))
        periodicity: torch.tensor(shape=(1, int(samples // hopsize)))
    """
    # Maybe use default output filenames
    if output_prefixes is None:
        output_prefixes = len(files) * [None]

    # Inference
    for file, output_prefix in zip(files, output_prefixes):
        from_file_to_file(
            file,
            output_prefix,
            hopsize,
            fmin,
            fmax,
            model,
            checkpoint,
            batch_size,
            device)


###############################################################################
# Utilities
###############################################################################


def iterator(iterable, message, length=None):
    """Create a tqdm iterator"""
    length = len(iterable) if length is None else length
    return tqdm.tqdm(
        iterable,
        desc=message,
        dynamic_ncols=True,
        total=length)


def resample(audio, sample_rate, target_rate=penne.SAMPLE_RATE):
    """Perform audio resampling"""
    if sample_rate == target_rate:
        return audio
    resampler = torchaudio.transforms.Resample(sample_rate, target_rate)
    resampler = resampler.to(audio.device)
    return resampler(audio)

