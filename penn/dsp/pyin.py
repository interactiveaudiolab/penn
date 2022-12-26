import functools
import multiprocessing as mp

import numpy as np
import torch

import penn


###############################################################################
# PYIN (from librosa)
###############################################################################


def from_audio(
    audio,
    sample_rate=penn.SAMPLE_RATE,
    hopsize=penn.HOPSIZE_SECONDS,
    fmin=penn.FMIN,
    fmax=penn.FMAX):
    """Estimate pitch and periodicity with pyin"""
    # Pad
    pad = int(
        penn.WINDOW_SIZE - penn.convert.seconds_to_samples(hopsize)) // 2
    audio = torch.nn.functional.pad(audio, (pad, pad))

    # Infer pitch bin probabilities
    with penn.time.timer('infer'):
        logits = infer(audio, sample_rate, hopsize, fmin, fmax)

    # Decode pitch and periodicity
    with penn.time.timer('postprocess'):
        return penn.postprocess(logits)[1:]


def from_file(
    file,
    hopsize=penn.HOPSIZE_SECONDS,
    fmin=penn.FMIN,
    fmax=penn.FMAX):
    """Estimate pitch and periodicity with pyin from audio on disk"""
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
    """Estimate pitch and periodicity with pyin and save to disk"""
    # Infer
    pitch, periodicity = from_file(file, hopsize, fmin, fmax)

    # Save to disk
    with penn.time.timer('save'):

        # Maybe use same filename with new extension
        if output_prefix is None:
            output_prefix = file.parent / file.stem

        # Save
        torch.save(pitch, f'{output_prefix}-pitch.pt')
        torch.save(periodicity, f'{output_prefix}-periodicity.pt')


def from_files_to_files(
    files,
    output_prefixes=None,
    hopsize=penn.HOPSIZE_SECONDS,
    fmin=penn.FMIN,
    fmax=penn.FMAX):
    """Estimate pitch and periodicity with pyin and save to disk"""
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


###############################################################################
# Utilities
###############################################################################


def cumulative_mean_normalized_difference(frames, min_period, max_period):
    import librosa

    a = np.fft.rfft(frames, 2 * penn.WINDOW_SIZE, axis=-2)
    b = np.fft.rfft(
        frames[..., penn.WINDOW_SIZE:0:-1, :],
        2 * penn.WINDOW_SIZE,
        axis=-2)
    acf_frames = np.fft.irfft(
        a * b, 2 * penn.WINDOW_SIZE, axis=-2)[..., penn.WINDOW_SIZE:, :]
    acf_frames[np.abs(acf_frames) < 1e-6] = 0

    # Energy terms
    energy_frames = np.cumsum(frames ** 2, axis=-2)
    energy_frames = (
        energy_frames[..., penn.WINDOW_SIZE:, :] -
        energy_frames[..., :-penn.WINDOW_SIZE, :])
    energy_frames[np.abs(energy_frames) < 1e-6] = 0

    # Difference function
    yin_frames = energy_frames[..., :1, :] + energy_frames - 2 * acf_frames

    # Cumulative mean normalized difference function
    yin_numerator = yin_frames[..., min_period: max_period + 1, :]

    # Broadcast to have leading ones
    tau_range = librosa.util.expand_to(
        np.arange(1, max_period + 1), ndim=yin_frames.ndim, axes=-2)

    cumulative_mean = (
        np.cumsum(yin_frames[..., 1: max_period + 1, :], axis=-2) / tau_range)

    yin_denominator = cumulative_mean[..., min_period - 1: max_period, :]
    yin_frames = yin_numerator / \
        (yin_denominator + librosa.util.tiny(yin_denominator))

    return yin_frames


def infer(
    audio,
    sample_rate=penn.SAMPLE_RATE,
    hopsize=penn.HOPSIZE_SECONDS,
    fmin=penn.FMIN,
    fmax=penn.FMAX):
    hopsize = int(penn.convert.seconds_to_samples(hopsize))
    import scipy

    # Pad audio to center-align frames
    pad = penn.WINDOW_SIZE // 2
    padded = torch.nn.functional.pad(audio, (0, 2 * pad))

    # Slice and chunk audio
    frames = torch.nn.functional.unfold(
        padded[:, None, None],
        kernel_size=(1, 2 * penn.WINDOW_SIZE),
        stride=(1, penn.HOPSIZE))[0]

    # Calculate minimum and maximum periods
    min_period = max(int(np.floor(sample_rate / fmax)), 1)
    max_period = min(
        int(np.ceil(sample_rate / fmin)),
        penn.WINDOW_SIZE - 1)

    # Calculate cumulative mean normalized difference function
    yin_frames = cumulative_mean_normalized_difference(
        frames.numpy(),
        min_period,
        max_period)

    # Parabolic interpolation
    parabolic_shifts = parabolic_interpolation(yin_frames)

    # Find Yin candidates and probabilities.
    # The implementation here follows the official pYIN software which
    # differs from the method described in the paper.
    # 1. Define the prior over the thresholds.
    thresholds = np.linspace(0, 1, 100 + 1)
    beta_cdf = scipy.stats.beta.cdf(thresholds, 2, 18)
    beta_probs = np.diff(beta_cdf)

    def _helper(a, b):
        return pyin_helper(
            a,
            b,
            thresholds,
            2,
            beta_probs,
            .01,
            min_period,
            (penn.OCTAVE / 12) / penn.CENTS_PER_BIN)

    helper = np.vectorize(_helper, signature="(f,t),(k,t)->(1,d,t)")
    probs = helper(yin_frames, parabolic_shifts)
    probs = torch.from_numpy(probs)
    return torch.log(probs).permute(2, 1, 0)


def parabolic_interpolation(frames):
    """Piecewise parabolic interpolation for yin and pyin"""
    import librosa

    parabolic_shifts = np.zeros_like(frames)
    parabola_a = (
        frames[..., :-2, :] +
        frames[..., 2:, :] -
        2 * frames[..., 1:-1, :]
    ) / 2
    parabola_b = (frames[..., 2:, :] - frames[..., :-2, :]) / 2
    parabolic_shifts[..., 1:-1, :] = \
        -parabola_b / (2 * parabola_a + librosa.util.tiny(parabola_a))
    parabolic_shifts[np.abs(parabolic_shifts) > 1] = 0
    return parabolic_shifts


def pyin_helper(
        frames,
        parabolic_shifts,
        thresholds,
        boltzmann_parameter,
        beta_probs,
        no_trough_prob,
        min_period,
        n_bins_per_semitone):
    import librosa
    import scipy


    yin_probs = np.zeros_like(frames)

    for i, yin_frame in enumerate(frames.T):
        # 2. For each frame find the troughs.
        is_trough = librosa.util.localmin(yin_frame)

        is_trough[0] = yin_frame[0] < yin_frame[1]
        (trough_index,) = np.nonzero(is_trough)

        if len(trough_index) == 0:
            continue

        # 3. Find the troughs below each threshold.
        # these are the local minima of the frame, could get them directly
        # without the trough index
        trough_heights = yin_frame[trough_index]
        trough_thresholds = np.less.outer(trough_heights, thresholds[1:])

        # 4. Define the prior over the troughs.
        # Smaller periods are weighted more.
        trough_positions = np.cumsum(trough_thresholds, axis=0) - 1
        n_troughs = np.count_nonzero(trough_thresholds, axis=0)

        trough_prior = scipy.stats.boltzmann.pmf(
            trough_positions,
            boltzmann_parameter,
            n_troughs)

        trough_prior[~trough_thresholds] = 0

        # 5. For each threshold add probability to global minimum if no trough
        # is below threshold, else add probability to each trough below
        # threshold biased by prior.
        probs = trough_prior.dot(beta_probs)

        global_min = np.argmin(trough_heights)
        n_thresholds_below_min = np.count_nonzero(
            ~trough_thresholds[global_min, :])
        probs[global_min] += no_trough_prob * np.sum(
            beta_probs[:n_thresholds_below_min])

        yin_probs[trough_index, i] = probs

    yin_period, frame_index = np.nonzero(yin_probs)

    # Refine peak by parabolic interpolation.
    period_candidates = min_period + yin_period
    period_candidates = period_candidates + \
        parabolic_shifts[yin_period, frame_index]
    f0_candidates = penn.SAMPLE_RATE / period_candidates

    # Find pitch bin corresponding to each f0 candidate.
    bin_index = 12 * n_bins_per_semitone * np.log2(f0_candidates / penn.FMIN)
    bin_index = np.clip(
        np.round(bin_index),
        0,
        penn.PITCH_BINS - 1).astype(int)

    # Observation probabilities.
    observation_probs = np.zeros((penn.PITCH_BINS, frames.shape[1]))
    observation_probs[bin_index, frame_index] = \
        yin_probs[yin_period, frame_index]

    return observation_probs[np.newaxis]
