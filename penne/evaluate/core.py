import contextlib
import json
import math
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

import penne


###############################################################################
# Evaluate
###############################################################################


def datasets(
    datasets=penne.EVALUATION_DATASETS,
    checkpoint=penne.DEFAULT_CHECKPOINT,
    gpu=None):
    """Perform evaluation"""
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)

        # Evaluate pitch estimation quality and save logits
        pitch_quality(directory, datasets, checkpoint, gpu)

        # Get periodicity methods
        if penne.METHOD == 'dio':
            periodicity_fns = {}
        elif penne.METHOD == 'pyin':
            periodicity_fns = {'sum': penne.periodicity.sum}
        else:
            periodicity_fns = {
                'entropy': penne.periodicity.entropy,
                'max': penne.periodicity.max}

        # Use saved logits to further evaluate periodicity
        periodicity_results = {}
        for key, val in periodicity_fns.items():
            periodicity_results[key] = periodicity_quality(
                directory,
                val,
                datasets,
                gpu=gpu)

        # Write periodicity results
        file = penne.EVAL_DIR / penne.CONFIG / 'periodicity.json'
        with open(file, 'w') as file:
            json.dump(periodicity_results, file, indent=4)

    # Perform benchmarking on CPU
    # TEMPORARY - use all threads
    # with set_num_threads(1):
    benchmark_results = {'cpu': benchmark(datasets, checkpoint)}
    # benchmark_results = {'cpu': 0}

    # PYIN and DIO do not have GPU support
    if penne.METHOD not in ['dio', 'pyin']:
        benchmark_results ['gpu'] = benchmark(datasets, checkpoint, gpu)

    # Write benchmarking information
    with open(penne.EVAL_DIR / penne.CONFIG / 'time.json', 'w') as file:
        json.dump(benchmark_results, file, indent=4)



###############################################################################
# Individual evaluations
###############################################################################


def benchmark(
    datasets=penne.EVALUATION_DATASETS,
    checkpoint=penne.DEFAULT_CHECKPOINT,
    gpu=None):
    """Perform benchmarking"""
    # Get audio files
    dataset_stems = {
        dataset: penne.load.partition(dataset)['test'] for dataset in datasets}
    files = [
        penne.CACHE_DIR / dataset / f'{stem}.wav'
        for dataset, stems in dataset_stems.items()
        for stem in stems]

    # Setup temporary directory
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)

        # Create output directories
        for dataset in datasets:
            (directory / dataset).mkdir(exist_ok=True, parents=True)

        # Get output prefixes
        output_prefixes = [
            directory / file.parent.name / file.stem for file in files]

        # Start benchmarking
        penne.BENCHMARK = True
        penne.TIMER.reset()
        start_time = time.time()

        # Infer to temporary storage
        if penne.METHOD == 'penne':
            batch_size = \
                    None if gpu is None else penne.EVALUATION_BATCH_SIZE
            penne.from_files_to_files(
                files,
                output_prefixes,
                checkpoint=checkpoint,
                batch_size=batch_size,
                gpu=gpu)

        elif penne.METHOD == 'torchcrepe':

            import torchcrepe

            # Get output file paths
            pitch_files = [
                file.parent / f'{file.stem}-pitch.pt'
                for file in output_prefixes]
            periodicity_files = [
                file.parent / f'{file.stem}-periodicity.pt'
                for file in output_prefixes]

            # Infer
            # Note - this does not perform the correct padding, but suffices
            #        for benchmarking purposes
            batch_size = \
                    None if gpu is None else penne.EVALUATION_BATCH_SIZE
            torchcrepe.predict_from_files_to_files(
                files,
                pitch_files,
                output_periodicity_files=periodicity_files,
                hop_length=penne.HOPSIZE,
                decoder=torchcrepe.decoder.argmax,
                batch_size=batch_size,
                device='cpu' if gpu is None else f'cuda:{gpu}',
                pad=False)

        elif penne.METHOD == 'harmof0':
            penne.temp.harmof0.from_files_to_files(
                # TODO
            )
        elif penne.METHOD == 'dio':
            penne.dsp.dio.from_files_to_files(files, output_prefixes)
        elif penne.METHOD == 'pyin':
            penne.dsp.pyin.from_files_to_files(files, output_prefixes)

        # Turn off benchmarking
        penne.BENCHMARK = False

        # Get benchmarking information
        benchmark = penne.TIMER()
        benchmark['total'] = time.time() - start_time

    # Get total number of samples and seconds in test data
    samples = sum([
        len(np.load(file.parent / f'{file.stem}-audio.npy', mmap_mode='r'))
        for file in files])
    seconds = penne.convert.samples_to_seconds(samples)

    # Format benchmarking results
    return {
        key: {
            'real-time-factor': value / seconds,
            'samples': samples,
            'samples-per-second': samples / value,
            'seconds': value
        } for key, value in benchmark.items()}


def periodicity_quality(
    directory,
    periodicity_fn,
    datasets=penne.EVALUATION_DATASETS,
    steps=8,
    gpu=None):
    """Fine-grained periodicity estimation quality evaluation"""
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

    # Default values
    best_threshold = .5
    best_value = 0.
    stepsize = .05

    # Setup metrics
    metrics = penne.evaluate.metrics.F1()

    step = 0
    while step < steps:

        for dataset in datasets:

            # Iterate over test set
            for _, _, _, voiced, stem in penne.data.loader([dataset], 'test'):

                # Load logits
                logits = torch.load(directory / dataset / f'{stem}-logits.pt')

                # Decode periodicity
                periodicity = periodicity_fn(logits.to(device)).T

                # Update metrics
                metrics.update(periodicity, voiced.to(device))

        # Get best performing threshold
        results = {
            key: val for key, val in metrics().items() if key.startswith('f1')
            and not math.isnan(val)}
        key = max(results, key=results.get)
        threshold = float(key[3:])
        value = results[key]
        if value > best_value:
            best_value = value
            best_threshold = threshold

        # Reinitialize metrics with new thresholds
        metrics = penne.evaluate.metrics.F1(
            [best_threshold - stepsize, best_threshold + stepsize])

        # Binary search for optimal threshold
        stepsize /= 2
        step += 1

    # Return threshold and corresponding F1 score
    return {'threshold': best_threshold, 'f1': best_value}


def pitch_quality(
    directory,
    datasets=penne.EVALUATION_DATASETS,
    checkpoint=penne.DEFAULT_CHECKPOINT,
    gpu=None):
    """Evaluate pitch estimation quality"""
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

    # Containers for results
    overall, granular = {}, {}

    # Get metric class
    metric_fn = (
        penne.evaluate.PitchMetrics if penne.METHOD == 'dio' else
        penne.evaluate.Metrics)

    # Per-file metrics
    file_metrics = metric_fn()

    # Per-dataset metrics
    dataset_metrics = metric_fn()

    # Aggregate metrics over all datasets
    aggregate_metrics = metric_fn()

    # Evaluate each dataset
    for dataset in datasets:

        # Create output directory
        (directory / dataset).mkdir(exist_ok=True, parents=True)

        # Reset dataset metrics
        dataset_metrics.reset()

        # Setup test dataset
        iterator = penne.iterator(
            penne.data.loader([dataset], 'test'),
            f'Evaluating {penne.CONFIG} pitch quality on {dataset}')

        # Iterate over test set
        for audio, bins, pitch, voiced, stem in iterator:

            # Reset file metrics
            file_metrics.reset()

            if penne.METHOD == 'penne':

                # Accumulate logits
                logits = []

                # Preprocess audio
                batch_size = \
                    None if gpu is None else penne.EVALUATION_BATCH_SIZE
                iterator = penne.preprocess(
                    audio[0],
                    penne.SAMPLE_RATE,
                    model=penne.MODEL,
                    batch_size=batch_size)
                for i, (frames, size) in enumerate(iterator):

                    # Copy to device
                    frames = frames.to(device)

                    # Slice features and copy to GPU
                    start = i * penne.EVALUATION_BATCH_SIZE
                    end = start + size
                    batch_bins = bins[:, start:end].to(device)
                    batch_pitch = pitch[:, start:end].to(device)
                    batch_voiced = voiced[:, start:end].to(device)

                    # Infer
                    batch_logits = penne.infer(
                        frames,
                        penne.MODEL,
                        checkpoint).detach()

                    # Update metrics
                    args = (
                        batch_logits,
                        batch_bins,
                        batch_pitch,
                        batch_voiced)
                    file_metrics.update(*args)
                    dataset_metrics.update(*args)
                    aggregate_metrics.update(*args)

                    # Accumulate logits
                    logits.append(batch_logits)
                logits = torch.cat(logits)

            elif penne.METHOD == 'torchcrepe':

                import torchcrepe

                # Accumulate logits
                logits = []

                # Postprocessing breaks gradients, so just don't compute them
                with torch.no_grad():

                    # Preprocess audio
                    batch_size = \
                        None if gpu is None else penne.EVALUATION_BATCH_SIZE
                    pad = (penne.WINDOW_SIZE - penne.HOPSIZE) // 2
                    generator = torchcrepe.preprocess(
                        torch.nn.functional.pad(audio, (pad, pad))[0],
                        penne.SAMPLE_RATE,
                        penne.HOPSIZE,
                        batch_size,
                        device,
                        False)
                    for i, frames in enumerate(generator):

                        # Infer independent probabilities for each pitch bin
                        batch_logits = torchcrepe.infer(frames.to(device))[:, :, None]

                        # Slice features and copy to GPU
                        start = i * penne.EVALUATION_BATCH_SIZE
                        end = start + frames.shape[0]
                        batch_bins = bins[:, start:end].to(device)
                        batch_pitch = pitch[:, start:end].to(device)
                        batch_voiced = voiced[:, start:end].to(device)

                        # Update metrics
                        args = (
                            batch_logits,
                            batch_bins,
                            batch_pitch,
                            batch_voiced)
                        file_metrics.update(*args)
                        dataset_metrics.update(*args)
                        aggregate_metrics.update(*args)

                        # Accumulate logits
                        logits.append(batch_logits)
                    logits = torch.cat(logits)

            elif penne.METHOD == 'harmof0':

                # TODO
                pass

            elif penne.METHOD == 'dio':

                # Pad
                pad = (penne.WINDOW_SIZE - penne.HOPSIZE) // 2
                audio = torch.nn.functional.pad(audio, (pad, pad))

                # Infer
                predicted = penne.dsp.dio.from_audio(audio[0])

                import pdb; pdb.set_trace()

                # Update metrics
                args = predicted, pitch, voiced
                file_metrics.update(*args)
                dataset_metrics.update(*args)
                aggregate_metrics.update(*args)

            elif penne.METHOD == 'pyin':

                # Pad
                pad = (penne.WINDOW_SIZE - penne.HOPSIZE) // 2
                audio = torch.nn.functional.pad(audio, (pad, pad))

                # Infer
                logits = penne.dsp.pyin.infer(audio[0])

                # Update metrics
                args = logits, bins, pitch, voiced
                file_metrics.update(*args)
                dataset_metrics.update(*args)
                aggregate_metrics.update(*args)

            # Maybe save logits for periodicity evaluation
            if penne.METHOD != 'dio':
                file = directory / dataset / f'{stem}-logits.pt'
                torch.save(logits.cpu(), file)

            # Copy results
            granular[f'{dataset}/{stem[0]}'] = file_metrics()
        overall[dataset] = dataset_metrics()
    overall['aggregate'] = aggregate_metrics()

    # Make output directory
    directory = penne.EVAL_DIR / penne.CONFIG
    directory.mkdir(exist_ok=True, parents=True)

    # Write to json files
    with open(directory / 'overall.json', 'w') as file:
        json.dump(overall, file, indent=4)
    with open(directory / 'granular.json', 'w') as file:
        json.dump(granular, file, indent=4)


###############################################################################
# Utilities
###############################################################################


@contextlib.contextmanager
def set_num_threads(threads):
    """Set the number of threads used for PyTorch inference"""
    # Cache current thread count
    prev_threads = torch.get_num_threads()

    # Change count
    torch.set_num_threads(threads)

    # Execute user code
    yield

    # Change back
    torch.set_num_threads(prev_threads)

