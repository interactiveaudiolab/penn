import json
import math
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

import penn


###############################################################################
# Evaluate
###############################################################################


def datasets(
    datasets=penn.EVALUATION_DATASETS,
    checkpoint=penn.DEFAULT_CHECKPOINT,
    gpu=None):
    """Perform evaluation"""
    # Make output directory
    directory = penn.EVAL_DIR / penn.CONFIG
    directory.mkdir(exist_ok=True, parents=True)

    # Evaluate pitch estimation quality and save logits
    pitch_quality(directory, datasets, checkpoint, gpu)

    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)

        # Get periodicity methods
        if penn.METHOD == 'dio':
            periodicity_fns = {}
        elif penn.METHOD == 'pyin':
            periodicity_fns = {'sum': penn.periodicity.sum}
        else:
            periodicity_fns = {
                'entropy': penn.periodicity.entropy,
                'max': penn.periodicity.max}

        # Evaluate periodicity
        periodicity_results = {}
        for key, val in periodicity_fns.items():
            periodicity_results[key] = periodicity_quality(
                directory,
                val,
                datasets,
                checkpoint=checkpoint,
                gpu=gpu)

        # Write periodicity results
        file = penn.EVAL_DIR / penn.CONFIG / 'periodicity.json'
        with open(file, 'w') as file:
            json.dump(periodicity_results, file, indent=4)

    # Perform benchmarking on CPU
    benchmark_results = {'cpu': benchmark(datasets, checkpoint)}

    # PYIN and DIO do not have GPU support
    if penn.METHOD not in ['dio', 'pyin']:
        benchmark_results ['gpu'] = benchmark(datasets, checkpoint, gpu)

    # Write benchmarking information
    with open(penn.EVAL_DIR / penn.CONFIG / 'time.json', 'w') as file:
        json.dump(benchmark_results, file, indent=4)


###############################################################################
# Individual evaluations
###############################################################################


def benchmark(
    datasets=penn.EVALUATION_DATASETS,
    checkpoint=penn.DEFAULT_CHECKPOINT,
    gpu=None):
    """Perform benchmarking"""
    # Get audio files
    dataset_stems = {
        dataset: penn.load.partition(dataset)['test'] for dataset in datasets}
    files = [
        penn.CACHE_DIR / dataset / f'{stem}.wav'
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
        penn.BENCHMARK = True
        penn.TIMER.reset()
        start_time = time.time()

        # Infer to temporary storage
        if penn.METHOD == 'penn':
            batch_size = \
                    None if gpu is None else penn.EVALUATION_BATCH_SIZE
            penn.from_files_to_files(
                files,
                output_prefixes,
                checkpoint=checkpoint,
                batch_size=batch_size,
                pad=True,
                gpu=gpu)

        elif penn.METHOD == 'torchcrepe':

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
                    None if gpu is None else penn.EVALUATION_BATCH_SIZE
            torchcrepe.predict_from_files_to_files(
                files,
                pitch_files,
                output_periodicity_files=periodicity_files,
                hop_length=penn.HOPSIZE,
                decoder=torchcrepe.decode.argmax,
                batch_size=batch_size,
                device='cpu' if gpu is None else f'cuda:{gpu}',
                pad=False)
        elif penn.METHOD == 'dio':
            penn.dsp.dio.from_files_to_files(files, output_prefixes)
        elif penn.METHOD == 'pyin':
            penn.dsp.pyin.from_files_to_files(files, output_prefixes)

        # Turn off benchmarking
        penn.BENCHMARK = False

        # Get benchmarking information
        benchmark = penn.TIMER()
        benchmark['total'] = time.time() - start_time

    # Get total number of samples and seconds in test data
    samples = sum([
        len(np.load(file.parent / f'{file.stem}-audio.npy', mmap_mode='r'))
        for file in files])
    seconds = penn.convert.samples_to_seconds(samples)

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
        datasets=penn.EVALUATION_DATASETS,
        steps=8,
        checkpoint=penn.DEFAULT_CHECKPOINT,
        gpu=None):
    """Fine-grained periodicity estimation quality evaluation"""
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

    # Evaluate each dataset
    for dataset in datasets:

        # Create output directory
        (directory / dataset).mkdir(exist_ok=True, parents=True)

        # Setup dataset
        iterator = penn.iterator(
            penn.data.loader([dataset], 'valid', gpu, True),
            f'Evaluating {penn.CONFIG} periodicity quality on {dataset}')

        # Iterate over validation set
        for audio, _, _, voiced, stem in iterator:

            if penn.METHOD == 'penn':

                # Accumulate logits
                logits = []

                # Preprocess audio
                batch_size = \
                    None if gpu is None else penn.EVALUATION_BATCH_SIZE
                iterator = penn.preprocess(
                    audio[0],
                    penn.SAMPLE_RATE,
                    batch_size=batch_size,
                    pad=True)
                for frames, _ in iterator:

                    # Copy to device
                    frames = frames.to(device)

                    # Infer
                    batch_logits = penn.infer(frames, checkpoint).detach()

                    # Accumulate logits
                    logits.append(batch_logits)

                logits = torch.cat(logits)

            elif penn.METHOD == 'torchcrepe':

                import torchcrepe

                # Accumulate logits
                logits = []

                # Postprocessing breaks gradients, so just don't compute them
                with torch.no_grad():

                    # Preprocess audio
                    batch_size = \
                        None if gpu is None else penn.EVALUATION_BATCH_SIZE
                    pad = (penn.WINDOW_SIZE - penn.HOPSIZE) // 2
                    generator = torchcrepe.preprocess(
                        torch.nn.functional.pad(audio, (pad, pad))[0],
                        penn.SAMPLE_RATE,
                        penn.HOPSIZE,
                        batch_size,
                        device,
                        False)
                    for frames in generator:

                        # Infer independent probabilities for each pitch bin
                        batch_logits = torchcrepe.infer(
                            frames.to(device))[:, :, None]

                        # Accumulate logits
                        logits.append(batch_logits)
                    logits = torch.cat(logits)

            elif penn.METHOD == 'pyin':

                # Pad
                pad = (penn.WINDOW_SIZE - penn.HOPSIZE) // 2
                audio = torch.nn.functional.pad(audio, (pad, pad))

                # Infer
                logits = penn.dsp.pyin.infer(audio[0])

            # Save to temporary storage
            file = directory / dataset / f'{stem[0]}-logits.pt'
            torch.save(logits, file)

    # Default values
    best_threshold = .5
    best_value = 0.
    stepsize = .05

    # Setup metrics
    metrics = penn.evaluate.metrics.F1()

    step = 0
    while step < steps:

        for dataset in datasets:

            # Setup loader
            loader = penn.data.loader([dataset], 'valid', gpu, True)

            # Iterate over validation set
            for _, _, _, voiced, stem in loader:

                # Load logits
                logits = torch.load(directory / dataset / f'{stem[0]}-logits.pt')

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
        metrics = penn.evaluate.metrics.F1(
            [best_threshold - stepsize, best_threshold + stepsize])

        # Binary search for optimal threshold
        stepsize /= 2
        step += 1

    # Setup metrics with optimal threshold
    metrics = penn.evaluate.metrics.F1([best_threshold])

    # Setup test loader
    loader = penn.data.loader(datasets, 'test', gpu)

    # Iterate over test set
    for audio, _, _, voiced, stem in loader:

        if penn.METHOD == 'penn':

            # Accumulate logits
            logits = []

            # Preprocess audio
            batch_size = \
                None if gpu is None else penn.EVALUATION_BATCH_SIZE
            iterator = penn.preprocess(
                audio[0],
                penn.SAMPLE_RATE,
                batch_size=batch_size,
                pad=True)
            for frames, _ in iterator:

                # Copy to device
                frames = frames.to(device)

                # Infer
                batch_logits = penn.infer(frames, checkpoint).detach()

                # Accumulate logits
                logits.append(batch_logits)

            logits = torch.cat(logits)

        elif penn.METHOD == 'torchcrepe':

            import torchcrepe

            # Accumulate logits
            logits = []

            # Postprocessing breaks gradients, so just don't compute them
            with torch.no_grad():

                # Preprocess audio
                batch_size = \
                    None if gpu is None else penn.EVALUATION_BATCH_SIZE
                pad = (penn.WINDOW_SIZE - penn.HOPSIZE) // 2
                generator = torchcrepe.preprocess(
                    torch.nn.functional.pad(audio, (pad, pad))[0],
                    penn.SAMPLE_RATE,
                    penn.HOPSIZE,
                    batch_size,
                    device,
                    False)
                for frames in generator:

                    # Infer independent probabilities for each pitch bin
                    batch_logits = torchcrepe.infer(
                        frames.to(device))[:, :, None]

                    # Accumulate logits
                    logits.append(batch_logits)
                logits = torch.cat(logits)

        elif penn.METHOD == 'pyin':

            # Pad
            pad = (penn.WINDOW_SIZE - penn.HOPSIZE) // 2
            audio = torch.nn.functional.pad(audio, (pad, pad))

            # Infer
            logits = penn.dsp.pyin.infer(audio[0]).to(device)

        # Decode periodicity
        periodicity = periodicity_fn(logits).T

        # Update metrics
        metrics.update(periodicity, voiced.to(device))

    # Get F1 score on test set
    score = metrics()[f'f1-{best_threshold:.6f}']

    return {'threshold': best_threshold, 'f1': score}


def pitch_quality(
    directory,
    datasets=penn.EVALUATION_DATASETS,
    checkpoint=penn.DEFAULT_CHECKPOINT,
    gpu=None):
    """Evaluate pitch estimation quality"""
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

    # Containers for results
    overall, granular = {}, {}

    # Get metric class
    metric_fn = (
        penn.evaluate.PitchMetrics if penn.METHOD == 'dio' else
        penn.evaluate.Metrics)

    # Per-file metrics
    file_metrics = metric_fn()

    # Per-dataset metrics
    dataset_metrics = metric_fn()

    # Aggregate metrics over all datasets
    aggregate_metrics = metric_fn()

    # Evaluate each dataset
    for dataset in datasets:

        # Reset dataset metrics
        dataset_metrics.reset()

        # Setup test dataset
        iterator = penn.iterator(
            penn.data.loader([dataset], 'test'),
            f'Evaluating {penn.CONFIG} pitch quality on {dataset}')

        # Iterate over test set
        for audio, bins, pitch, voiced, stem in iterator:

            # Reset file metrics
            file_metrics.reset()

            if penn.METHOD == 'penn':

                # Accumulate logits
                logits = []

                # Preprocess audio
                batch_size = \
                    None if gpu is None else penn.EVALUATION_BATCH_SIZE
                iterator = penn.preprocess(
                    audio[0],
                    penn.SAMPLE_RATE,
                    batch_size=batch_size,
                    pad=True)
                for i, (frames, size) in enumerate(iterator):

                    # Copy to device
                    frames = frames.to(device)

                    # Slice features and copy to GPU
                    start = i * penn.EVALUATION_BATCH_SIZE
                    end = start + size
                    batch_bins = bins[:, start:end].to(device)
                    batch_pitch = pitch[:, start:end].to(device)
                    batch_voiced = voiced[:, start:end].to(device)

                    # Infer
                    batch_logits = penn.infer(frames, checkpoint).detach()

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

            elif penn.METHOD == 'torchcrepe':

                import torchcrepe

                # Accumulate logits
                logits = []

                # Postprocessing breaks gradients, so just don't compute them
                with torch.no_grad():

                    # Preprocess audio
                    batch_size = \
                        None if gpu is None else penn.EVALUATION_BATCH_SIZE
                    pad = (penn.WINDOW_SIZE - penn.HOPSIZE) // 2
                    generator = torchcrepe.preprocess(
                        torch.nn.functional.pad(audio, (pad, pad))[0],
                        penn.SAMPLE_RATE,
                        penn.HOPSIZE,
                        batch_size,
                        device,
                        False)
                    for i, frames in enumerate(generator):

                        # Infer independent probabilities for each pitch bin
                        batch_logits = torchcrepe.infer(frames.to(device))[:, :, None]

                        # Slice features and copy to GPU
                        start = i * penn.EVALUATION_BATCH_SIZE
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

            elif penn.METHOD == 'dio':

                # Pad
                pad = (penn.WINDOW_SIZE - penn.HOPSIZE) // 2
                audio = torch.nn.functional.pad(audio, (pad, pad))

                # Infer
                predicted = penn.dsp.dio.from_audio(audio[0])

                # Update metrics
                args = predicted, pitch, voiced
                file_metrics.update(*args)
                dataset_metrics.update(*args)
                aggregate_metrics.update(*args)

            elif penn.METHOD == 'pyin':

                # Pad
                pad = (penn.WINDOW_SIZE - penn.HOPSIZE) // 2
                audio = torch.nn.functional.pad(audio, (pad, pad))

                # Infer
                logits = penn.dsp.pyin.infer(audio[0])

                # Update metrics
                args = logits, bins, pitch, voiced
                file_metrics.update(*args)
                dataset_metrics.update(*args)
                aggregate_metrics.update(*args)

            # Copy results
            granular[f'{dataset}/{stem[0]}'] = file_metrics()
        overall[dataset] = dataset_metrics()
    overall['aggregate'] = aggregate_metrics()

    # Write to json files
    directory = penn.EVAL_DIR / penn.CONFIG
    with open(directory / 'overall.json', 'w') as file:
        json.dump(overall, file, indent=4)
    with open(directory / 'granular.json', 'w') as file:
        json.dump(granular, file, indent=4)
