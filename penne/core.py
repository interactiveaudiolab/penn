import contextlib
import os
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
    fmax: float = penne.FMAX,
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
    for frames, _ in iterator:

        # Copy to device
        with penne.time.timer('copy-to'):
            frames = frames.to('cpu' if gpu is None else f'cuda:{gpu}')

        # Infer
        logits = infer(frames, model, checkpoint).detach()

        # Postprocess
        with penne.time.timer('postprocess'):
            result = postprocess(logits, fmin, fmax)
            pitch.append(result[1])
            periodicity.append(result[2])

    # Concatenate results
    return torch.cat(pitch, 1), torch.cat(periodicity, 1)


def from_file(
    file: Path,
    hopsize: float = penne.HOPSIZE_SECONDS,
    fmin: float = penne.FMIN,
    fmax: float = penne.FMAX,
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
    with penne.time.timer('load'):
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
    fmax: float = penne.FMAX,
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

    # Move to cpu
    with penne.time.timer('copy-from'):
        pitch, periodicity = pitch.cpu(), periodicity.cpu()

    # Save to disk
    with penne.time.timer('save'):

        # Maybe use same filename with new extension
        if output_prefix is None:
            output_prefix = file.parent / file.stem

        # Save
        torch.save(pitch, f'{output_prefix}-pitch.pt')
        torch.save(periodicity, f'{output_prefix}-periodicity.pt')


def from_files_to_files(
    files: Path,
    output_prefixes: Optional[list] = None,
    hopsize: float = penne.HOPSIZE_SECONDS,
    fmin: float = penne.FMIN,
    fmax: float = penne.FMAX,
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
    """
    # Maybe use default output filenames
    if output_prefixes is None:
        output_prefixes = len(files) * [None]

    # Iterate over files
    for file, output_prefix in iterator(
        zip(files, output_prefixes),
        f'{penne.CONFIG}',
        total=len(files)):

        # Infer
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
# Inference pipeline stages
###############################################################################


def infer(
    frames,
    model=penne.MODEL,
    checkpoint=penne.DEFAULT_CHECKPOINT):
    """Forward pass through the model"""
    # Time model loading
    with penne.time.timer('model'):

        # Load and cache model
        if (
            not hasattr(infer, 'model') or
            infer.checkpoint != checkpoint or
            infer.device_type != frames.device.type
        ):

            # Maybe initialize model
            if penne.ONNX and frames.device.type == 'cpu':
                model = None
            else:
                model = penne.Model(model)

            # Load from disk
            infer.model, *_ = penne.checkpoint.load(checkpoint, model)
            infer.checkpoint = checkpoint
            infer.device_type = frames.device.type

            # Move model to correct device (no-op if devices are the same)
            if not penne.ONNX or frames.device.type == 'cuda':
                infer.model = infer.model.to(frames.device)

    # Time inference
    with penne.time.timer('infer'):

        if penne.ONNX and frames.device.type == 'cpu':

            # Infer
            logits = infer.model.run(
                None,
                {infer.model.get_inputs()[0].name: frames.numpy()})[0]
            logits = torch.from_numpy(logits)

        else:

            # Prepare model for inference
            with inference_context(infer.model):

                # Infer
                logits = infer.model(frames)

        # If we're benchmarking, make sure inference finishes within timer
        if penne.BENCHMARK and logits.device.type == 'cuda':
            torch.cuda.synchronize(logits.device)

        return logits


def postprocess(logits, fmin=penne.FMIN, fmax=penne.FMAX):
    """Convert model output to pitch and periodicity"""
    # Turn off gradients
    with torch.no_grad():

        # Convert frequency range to pitch bin range
        minidx = penne.convert.frequency_to_bins(torch.tensor(fmin))
        maxidx = penne.convert.frequency_to_bins(
            torch.tensor(fmax),
            torch.ceil)

        # Remove frequencies outside of allowable range
        logits[:, :minidx] = -float('inf')
        logits[:, maxidx:] = -float('inf')

        # Decode pitch from logits
        if penne.DECODER == 'argmax':
            bins, pitch = penne.decode.argmax(logits)
        elif penne.DECODER == 'average':
            bins, pitch = penne.decode.average(logits)
        elif penne.DECODER.startswith('viterbi'):
            bins, pitch = penne.decode.viterbi(logits)
        elif penne.DECODER == 'weighted':
            bins, pitch = penne.decode.weighted(logits)
        else:
            raise ValueError(f'Decoder method {penne.DECODER} is not defined')

        # Decode periodicity from logits
        if penne.PERIODICITY == 'entropy':
            periodicity = penne.periodicity.entropy(logits)
        elif penne.PERIODICITY == 'max':
            periodicity = penne.periodicity.max(logits)
        elif penne.PERIODICITY == 'sum':
            periodicity = penne.periodicity.sum(logits)
        else:
            raise ValueError(
                f'Periodicity method {penne.PERIODICITY} is not defined')

        return bins.T, pitch.T, periodicity.T


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

        if model != 'harmof0':

            # Slice and chunk audio
            frames = torch.nn.functional.unfold(
                audio[:, None, None, start:end],
                kernel_size=(1, penne.WINDOW_SIZE),
                stride=(1, hopsize)).permute(2, 0, 1)

            yield frames, batch

        elif model == 'harmof0':

            # Slice audio
            yield audio[:, None, start:end], batch


###############################################################################
# Utilities
###############################################################################


@contextlib.contextmanager
def chdir(directory):
    """Context manager for changing the current working directory"""
    curr_dir = os.getcwd()
    try:
        os.chdir(directory)
        yield
    finally:
        os.chdir(curr_dir)


@contextlib.contextmanager
def inference_context(model):
    device_type = next(model.parameters()).device.type

    # Prepare model for evaluation
    model.eval()

    # Turn off gradient computation
    with torch.no_grad():

        # Automatic mixed precision on GPU
        if device_type == 'cuda':
            with torch.autocast(device_type):
                yield

        else:
            yield

    # Prepare model for training
    model.train()


def iterator(iterable, message, initial=0, total=None):
    """Create a tqdm iterator"""
    total = len(iterable) if total is None else total
    return tqdm.tqdm(
        iterable,
        desc=message,
        dynamic_ncols=True,
        initial=initial,
        total=total)


def normalize(frames):
    """Normalize audio frames to have mean zero and std dev one"""
    # Mean-center
    frames -= frames.mean(dim=2, keepdim=True)

    # Scale
    frames /= torch.max(
        torch.tensor(1e-10, device=frames.device),
        frames.std(dim=2, keepdim=True))

    return frames


def resample(audio, sample_rate, target_rate=penne.SAMPLE_RATE):
    """Perform audio resampling"""
    if sample_rate == target_rate:
        return audio
    resampler = torchaudio.transforms.Resample(sample_rate, target_rate)
    resampler = resampler.to(audio.device)
    return resampler(audio)
