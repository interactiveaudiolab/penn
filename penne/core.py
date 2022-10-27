from pathlib import Path
from typing import Optional, Tuple

import torch
import torchaudio
import tqdm

import penne


###############################################################################
# Pitch and periodicity estimation
###############################################################################


def from_audio(
    audio: torch.Tensor,
    sample_rate: int,
    hopsize: float = penne.HOPSIZE_SECONDS,
    fmin: float = penne.FMIN,
    fmax: Optional[float] = None,
    model: str = penne.MODEL,
    checkpoint: Path = penne.DEFAULT_CHECKPOINT,
    batch_size: Optional[int] = None,
    gpu: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
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
        gpu: The index of the gpu to run inference on

    Returns:
        pitch: torch.tensor(shape=(1, int(samples // hopsize)))
        periodicity: torch.tensor(shape=(1, int(samples // hopsize)))
    """
    pitch, periodicity = [], []

    # Preprocess audio
    iterator = preprocess(
        audio,
        sample_rate,
        hopsize,
        model,
        batch_size)
    for frames in iterator:

        # Copy to device
        frames = frames.to('cpu' if gpu is None else f'cuda:{gpu}')

        # Infer
        logits = infer(frames, model, checkpoint).detach()

        # Decode
        result = decode(logits, fmin, fmax)
        pitch.append(result[0])
        periodicity.append(result[1])

    # Concatenate results
    return torch.cat(pitch, 1), torch.cat(periodicity, 1)


def from_file(
    file: Path,
    hopsize: float = penne.HOPSIZE_SECONDS,
    fmin: float = penne.FMIN,
    fmax: Optional[float] = None,
    model: str = penne.MODEL,
    checkpoint: Path = penne.DEFAULT_CHECKPOINT,
    batch_size: Optional[int] = None,
    gpu: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perform pitch and periodicity estimation from audio on disk

    Args:
        file: The audio file
        hopsize: The hopsize in seconds
        fmin: The minimum allowable frequency in Hz
        fmax: The maximum allowable frequency in Hz
        model: The name of the model
        checkpoint: The checkpoint file
        batch_size: The number of frames per batch
        gpu: The index of the gpu to run inference on

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
        gpu)


def from_file_to_file(
    file: Path,
    output_prefix: Optional[Path] = None,
    hopsize: float = penne.HOPSIZE_SECONDS,
    fmin: float = penne.FMIN,
    fmax: Optional[float] = None,
    model: str = penne.MODEL,
    checkpoint: Path = penne.DEFAULT_CHECKPOINT,
    batch_size: Optional[int] = None,
    gpu: Optional[int] = None) -> None:
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
        gpu: The index of the gpu to run inference on

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
        gpu)

    # Maybe use same filename with new extension
    if output_prefix is None:
        output_prefix = file.parent / file.stem

    # Save to disk
    torch.save(pitch, f'{output_prefix}-pitch.pt')
    torch.save(periodicity, f'{output_prefix}-periodicity.pt')


def from_files_to_files(
    files: Path,
    output_prefixes: Optional[list] = None,
    hopsize: float = penne.HOPSIZE_SECONDS,
    fmin: float = penne.FMIN,
    fmax: Optional[float] = None,
    model: str = penne.MODEL,
    checkpoint: Path = penne.DEFAULT_CHECKPOINT,
    batch_size: Optional[int] = None,
    gpu: Optional[int] = None) -> None:
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
        gpu: The index of the gpu to run inference on

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
            gpu)


###############################################################################
# Utilities
###############################################################################


def infer(
    frames,
    model=penne.MODEL,
    checkpoint=penne.DEFAULT_CHECKPOINT):
    """Forward pass through the model"""
    # Time model loading
    with penne.time.timer('model'):

        # Load and cache model
        if not hasattr(infer, 'model') or infer.checkpoint != checkpoint:
            infer.model, *_ = penne.checkpoint.load(
                checkpoint,
                penne.Model(model))
            infer.checkpoint = checkpoint

            # Move model to correct device (no-op if devices are the same)
            infer.model = infer.model.to(frames.device)

            # Evaluation mode
            infer.model.eval()

            # Maybe use torchscript
            if penne.TORCHSCRIPT:
                infer.model = torch.jit.trace(infer.model, frames)

        else:

            # Move model to correct device (no-op if devices are the same)
            infer.model = infer.model.to(frames.device)

    # Time inference
    with penne.time.timer('infer'):

        # Turn off gradients
        with torch.no_grad():

            # Apply model
            logits = infer.model(frames)

        # Maybe reshape
        if model == 'crepe':
            logits = logits.permute(2, 1, 0)

        return logits


def decode(logits, fmin=penne.FMIN, fmax=None):
    """Convert model output to pitch and periodicity"""
    # Convert frequency range to pitch bin range
    minidx = penne.convert.frequency_to_bins(torch.tensor(fmin))
    if fmax is None:
        maxidx = penne.PITCH_BINS
    else:
        maxidx = penne.convert.frequency_to_bins(
            torch.tensor(fmax),
            torch.ceil)

    # Remove frequencies outside of allowable range
    logits[:, :minidx] = -float('inf')
    logits[:, maxidx:] = -float('inf')

    # Get pitch bins
    bins = logits.argmax(dim=1)

    # Convert to hz
    pitch = penne.convert.bins_to_frequency(bins)

    # Get periodicity
    if penne.PERIODICITY == 'entropy':
        periodicity = penne.periodicity.entropy(logits)
    elif penne.PERIODICITY == 'max':
        periodicity = penne.periodicity.max(logits)
    else:
        raise ValueError(
            f'Periodicity method {penne.PERIODICITY} is not defined')

    return pitch, periodicity


def preprocess(audio,
               sample_rate,
               hopsize=penne.HOPSIZE_SECONDS,
               model=penne.MODEL,
               batch_size=None):
    """Convert audio to model input"""
    # Convert hopsize to samples
    hopsize = penne.convert.seconds_to_samples(hopsize)

    # Resample
    if sample_rate != penne.SAMPLE_RATE:
        audio = resample(audio, sample_rate)

    # Get total number of frames
    total_frames = int(audio.shape[-1] / hopsize)

    # Pad audio
    padding = int((penne.WINDOW_SIZE - hopsize) / 2)
    audio = torch.nn.functional.pad(audio, (padding, padding))

    # Default to running all frames in a single batch
    batch_size = total_frames if batch_size is None else batch_size

    # Generate batches
    for i in range(0, total_frames, batch_size):

        # Size of this batch
        batch = min(total_frames - i, batch_size)

        # Batch indices
        start = i * hopsize
        end = start + int((batch - 1) * hopsize) + penne.WINDOW_SIZE

        if model == 'crepe':

            # Slice and chunk audio
            yield torch.nn.functional.unfold(
                audio[:, None, None, start:end],
                kernel_size=(1, penne.WINDOW_SIZE),
                stride=(1, hopsize)).permute(2, 0, 1)

        elif model == 'harmof0':

            # Slice audio
            yield audio[:, None, start:end]


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
