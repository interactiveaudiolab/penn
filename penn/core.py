import contextlib
import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torchaudio
import tqdm

import penn


###############################################################################
# Pitch and periodicity estimation
###############################################################################


def from_audio(
        audio: torch.Tensor,
        sample_rate: int = penn.SAMPLE_RATE,
        hopsize: float = penn.HOPSIZE_SECONDS,
        fmin: float = penn.FMIN,
        fmax: float = penn.FMAX,
        checkpoint: Path = penn.DEFAULT_CHECKPOINT,
        batch_size: Optional[int] = None,
        center: str = 'half-frame',
        interp_unvoiced_at: Optional[float] = None,
        gpu: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perform pitch and periodicity estimation

    Args:
        audio: The audio to extract pitch and periodicity from
        sample_rate: The audio sample rate
        hopsize: The hopsize in seconds
        fmin: The minimum allowable frequency in Hz
        fmax: The maximum allowable frequency in Hz
        checkpoint: The checkpoint file
        batch_size: The number of frames per batch
        center: Padding options. One of ['half-frame', 'half-hop', 'zero'].
        interp_unvoiced_at: Specifies voicing threshold for interpolation
        gpu: The index of the gpu to run inference on

    Returns:
        pitch: torch.tensor(
            shape=(1, int(samples // penn.seconds_to_sample(hopsize))))
        periodicity: torch.tensor(
            shape=(1, int(samples // penn.seconds_to_sample(hopsize))))
    """
    pitch, periodicity = [], []

    # Preprocess audio
    for frames, _ in preprocess(
        audio,
        sample_rate,
        hopsize,
        batch_size,
        center
    ):

        # Copy to device
        with penn.time.timer('copy-to'):
            frames = frames.to('cpu' if gpu is None else f'cuda:{gpu}')

        # Infer
        logits = infer(frames, checkpoint).detach()

        # Postprocess
        with penn.time.timer('postprocess'):
            result = postprocess(logits, fmin, fmax)
            pitch.append(result[1])
            periodicity.append(result[2])

    # Concatenate results
    pitch, periodicity = torch.cat(pitch, 1), torch.cat(periodicity, 1)

    # Maybe interpolate unvoiced regions
    if interp_unvoiced_at is not None:
        pitch = penn.voicing.interpolate(
            pitch,
            periodicity,
            interp_unvoiced_at)

    return pitch, periodicity


def from_file(
        file: Path,
        hopsize: float = penn.HOPSIZE_SECONDS,
        fmin: float = penn.FMIN,
        fmax: float = penn.FMAX,
        checkpoint: Path = penn.DEFAULT_CHECKPOINT,
        batch_size: Optional[int] = None,
        center: str = 'half-frame',
        interp_unvoiced_at: Optional[float] = None,
        gpu: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perform pitch and periodicity estimation from audio on disk

    Args:
        file: The audio file
        hopsize: The hopsize in seconds
        fmin: The minimum allowable frequency in Hz
        fmax: The maximum allowable frequency in Hz
        checkpoint: The checkpoint file
        batch_size: The number of frames per batch
        center: Padding options. One of ['half-frame', 'half-hop', 'zero'].
        interp_unvoiced_at: Specifies voicing threshold for interpolation
        gpu: The index of the gpu to run inference on

    Returns:
        pitch: torch.tensor(shape=(1, int(samples // hopsize)))
        periodicity: torch.tensor(shape=(1, int(samples // hopsize)))
    """
    # Load audio
    with penn.time.timer('load'):
        audio, sample_rate = torchaudio.load(file)

    # Inference
    return from_audio(
        audio,
        sample_rate,
        hopsize,
        fmin,
        fmax,
        checkpoint,
        batch_size,
        center,
        interp_unvoiced_at,
        gpu)


def from_file_to_file(
        file: Path,
        output_prefix: Optional[Path] = None,
        hopsize: float = penn.HOPSIZE_SECONDS,
        fmin: float = penn.FMIN,
        fmax: float = penn.FMAX,
        checkpoint: Path = penn.DEFAULT_CHECKPOINT,
        batch_size: Optional[int] = None,
        center: str = 'half-frame',
        interp_unvoiced_at: Optional[float] = None,
        gpu: Optional[int] = None) -> None:
    """Perform pitch and periodicity estimation from audio on disk and save

    Args:
        file: The audio file
        output_prefix: The file to save pitch and periodicity without extension
        hopsize: The hopsize in seconds
        fmin: The minimum allowable frequency in Hz
        fmax: The maximum allowable frequency in Hz
        checkpoint: The checkpoint file
        batch_size: The number of frames per batch
        center: Padding options. One of ['half-frame', 'half-hop', 'zero'].
        interp_unvoiced_at: Specifies voicing threshold for interpolation
        gpu: The index of the gpu to run inference on
    """
    # Inference
    pitch, periodicity = from_file(
        file,
        hopsize,
        fmin,
        fmax,
        checkpoint,
        batch_size,
        center,
        interp_unvoiced_at,
        gpu)

    # Move to cpu
    with penn.time.timer('copy-from'):
        pitch, periodicity = pitch.cpu(), periodicity.cpu()

    # Save to disk
    with penn.time.timer('save'):

        # Maybe use same filename with new extension
        if output_prefix is None:
            output_prefix = file.parent / file.stem

        # Save
        torch.save(pitch, f'{output_prefix}-pitch.pt')
        torch.save(periodicity, f'{output_prefix}-periodicity.pt')


def from_files_to_files(
    files: List[Path],
    output_prefixes: Optional[List[Path]] = None,
    hopsize: float = penn.HOPSIZE_SECONDS,
    fmin: float = penn.FMIN,
    fmax: float = penn.FMAX,
    checkpoint: Path = penn.DEFAULT_CHECKPOINT,
    batch_size: Optional[int] = None,
    center: str = 'half-frame',
    interp_unvoiced_at: Optional[float] = None,
    gpu: Optional[int] = None) -> None:
    """Perform pitch and periodicity estimation from files on disk and save

    Args:
        files: The audio files
        output_prefixes: Files to save pitch and periodicity without extension
        hopsize: The hopsize in seconds
        fmin: The minimum allowable frequency in Hz
        fmax: The maximum allowable frequency in Hz
        checkpoint: The checkpoint file
        batch_size: The number of frames per batch
        center: Padding options. One of ['half-frame', 'half-hop', 'zero'].
        interp_unvoiced_at: Specifies voicing threshold for interpolation
        gpu: The index of the gpu to run inference on
    """
    # Maybe use default output filenames
    if output_prefixes is None:
        output_prefixes = len(files) * [None]

    # Iterate over files
    for file, output_prefix in iterator(
        zip(files, output_prefixes),
        f'{penn.CONFIG}',
        total=len(files)):

        # Infer
        from_file_to_file(
            file,
            output_prefix,
            hopsize,
            fmin,
            fmax,
            checkpoint,
            batch_size,
            center,
            interp_unvoiced_at,
            gpu)


###############################################################################
# Inference pipeline stages
###############################################################################


def infer(frames, checkpoint=penn.DEFAULT_CHECKPOINT):
    """Forward pass through the model"""
    # Time model loading
    with penn.time.timer('model'):

        # Load and cache model
        if (
            not hasattr(infer, 'model') or
            infer.checkpoint != checkpoint or
            infer.device_type != frames.device.type
        ):

            # Maybe initialize model
            model = penn.Model()

            # Load from disk
            infer.model, *_ = penn.checkpoint.load(checkpoint, model)
            infer.checkpoint = checkpoint
            infer.device_type = frames.device.type

            # Move model to correct device (no-op if devices are the same)
            infer.model = infer.model.to(frames.device)

    # Time inference
    with penn.time.timer('infer'):

        # Prepare model for inference
        with inference_context(infer.model):

            # Infer
            logits = infer.model(frames)

        # If we're benchmarking, make sure inference finishes within timer
        if penn.BENCHMARK and logits.device.type == 'cuda':
            torch.cuda.synchronize(logits.device)

        return logits


def postprocess(logits, fmin=penn.FMIN, fmax=penn.FMAX):
    """Convert model output to pitch and periodicity"""
    # Turn off gradients
    with torch.no_grad():

        # Convert frequency range to pitch bin range
        minidx = penn.convert.frequency_to_bins(torch.tensor(fmin))
        maxidx = penn.convert.frequency_to_bins(
            torch.tensor(fmax),
            torch.ceil)

        # Remove frequencies outside of allowable range
        logits[:, :minidx] = -float('inf')
        logits[:, maxidx:] = -float('inf')

        # Decode pitch from logits
        if penn.DECODER == 'argmax':
            bins, pitch = penn.decode.argmax(logits)
        elif penn.DECODER.startswith('viterbi'):
            bins, pitch = penn.decode.viterbi(logits)
        elif penn.DECODER == 'local_expected_value':
            bins, pitch = penn.decode.local_expected_value(logits)
        else:
            raise ValueError(f'Decoder method {penn.DECODER} is not defined')

        # Decode periodicity from logits
        if penn.PERIODICITY == 'entropy':
            periodicity = penn.periodicity.entropy(logits)
        elif penn.PERIODICITY == 'max':
            periodicity = penn.periodicity.max(logits)
        elif penn.PERIODICITY == 'sum':
            periodicity = penn.periodicity.sum(logits)
        else:
            raise ValueError(
                f'Periodicity method {penn.PERIODICITY} is not defined')

        return bins.T, pitch.T, periodicity.T


def preprocess(
    audio,
    sample_rate=penn.SAMPLE_RATE,
    hopsize=penn.HOPSIZE_SECONDS,
    batch_size=None,
    center='half-frame'):
    """Convert audio to model input"""
    # Calculate expected number of frames
    hopsize_resampled = penn.convert.seconds_to_samples(
        hopsize,
        sample_rate)
    if center == 'half-frame':
        window_size_resampled = \
            penn.WINDOW_SIZE / penn.SAMPLE_RATE * sample_rate
        samples = audio.shape[-1] - (window_size_resampled - hopsize_resampled)
    elif center == 'half-hop':
        samples = audio.shape[-1]
    elif center == 'zero':
        samples = audio.shape[-1] + hopsize_resampled
    else:
        raise ValueError(f'Unknown center sample {center}')
    total_frames = max(1, int(samples / hopsize_resampled))

    # Maybe resample
    if sample_rate != penn.SAMPLE_RATE:
        audio = resample(audio, sample_rate)

    # Maybe pad audio
    hopsize = penn.convert.seconds_to_samples(hopsize)
    if center in ['half-hop', 'zero']:
        if center == 'half-hop':
            padding = int((penn.WINDOW_SIZE - hopsize) / 2)
        else:
            padding = int(penn.WINDOW_SIZE / 2)
        audio = torch.nn.functional.pad(
            audio,
            (padding, padding),
            mode='reflect')

    # Integer hopsizes permit a speedup using torch.unfold
    if isinstance(hopsize, int) or hopsize.is_integer():
        hopsize = int(round(hopsize))
        start_idxs = None

    else:

        # Find start indices
        start_idxs = torch.round(
            torch.tensor([hopsize * i for i in range(total_frames + 1)])
        ).int()

    # Default to running all frames in a single batch
    batch_size = total_frames if batch_size is None else batch_size

    # Generate batches
    for i in range(0, total_frames, batch_size):

        # Size of this batch
        batch = min(total_frames - i, batch_size)

        # Fast implementation for integer hopsizes
        if start_idxs is None:

            # Batch indices
            start = i * hopsize
            end = start + int((batch - 1) * hopsize) + penn.WINDOW_SIZE
            end = min(end, audio.shape[-1])
            batch_audio = audio[:, start:end]

            # Maybe pad to a single frame
            if end - start < penn.WINDOW_SIZE:
                padding = penn.WINDOW_SIZE - (end - start)

                # Handle multiple of hopsize
                remainder = (end - start) % hopsize
                if remainder:
                    padding += end - start - hopsize

                # Pad
                batch_audio = torch.nn.functional.pad(
                    batch_audio,
                    (0, padding))

            # Slice and chunk audio
            frames = torch.nn.functional.unfold(
                batch_audio[:, None, None],
                kernel_size=(1, penn.WINDOW_SIZE),
                stride=(1, hopsize)).permute(2, 0, 1)

        # Slow implementation for floating-point hopsizes
        else:

            # Allocate frames
            frames = torch.zeros(batch, 1, penn.WINDOW_SIZE)

            # Fill each frame with a window starting at the start index
            for j in range(batch):
                start = start_idxs[i + j]
                end = min(start + penn.WINDOW_SIZE, audio.shape[-1])
                frames[j, :, : end - start] = audio[:, start:end]

        yield frames, batch


###############################################################################
# Utilities
###############################################################################


def cents(a, b):
    """Compute pitch difference in cents"""
    return penn.OCTAVE * torch.log2(a / b)


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


def interpolate(x, xp, fp):
    """1D linear interpolation for monotonically increasing sample points"""
    # Handle edge cases
    if xp.shape[-1] == 0:
        return x
    if xp.shape[-1] == 1:
        return torch.full(
            x.shape,
            fp.squeeze(),
            device=fp.device,
            dtype=fp.dtype)

    # Get slope and intercept using right-side first-differences
    m = (fp[:, 1:] - fp[:, :-1]) / (xp[:, 1:] - xp[:, :-1])
    b = fp[:, :-1] - (m.mul(xp[:, :-1]))

    # Get indices to sample slope and intercept
    indicies = torch.sum(torch.ge(x[:, :, None], xp[:, None, :]), -1) - 1
    indicies = torch.clamp(indicies, 0, m.shape[-1] - 1)
    line_idx = torch.linspace(
        0,
        indicies.shape[0],
        1,
        device=indicies.device).to(torch.long).expand(indicies.shape)

    # Interpolate
    return m[line_idx, indicies].mul(x) + b[line_idx, indicies]


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


def resample(audio, sample_rate, target_rate=penn.SAMPLE_RATE):
    """Perform audio resampling"""
    if sample_rate == target_rate:
        return audio
    resampler = torchaudio.transforms.Resample(sample_rate, target_rate)
    resampler = resampler.to(audio.device)
    return resampler(audio)
