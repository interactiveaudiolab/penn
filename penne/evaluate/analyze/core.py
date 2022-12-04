import csv
import json

import penne


def analyze(runs=None):
    """Analyze results and save to disk"""
    # Default is all runs in eval directory
    if runs is None:
        directories = [
            item for item in penne.EVAL_DIR.glob('*') if item.is_dir()]
    else:
        directories = [penne.EVAL_DIR / run for run in runs]

    # Make alphabetical
    directories = sorted(directories)

    # Initialize tables
    datasets = penne.EVALUATION_DATASETS + ['aggregate']
    tables = {'time': [], 'quality': {dataset: [] for dataset in datasets}}

    # Populate tables
    for directory in directories:

        # Add timing results
        tables['time'].append(time_results(directory))

        # Add quality results
        for dataset in datasets:
            tables['quality'][dataset].append(
                quality_results(directory, dataset))

    # Save time table
    with open(penne.RESULTS_DIR / 'time.csv', 'w') as file:

        # Setup csv writer
        writer = csv.DictWriter(
            file,
            fieldnames=['name', 'rtf-cpu', 'rtf-gpu'],
            delimiter=',')

        # Write to csv
        writer.writeheader()
        writer.writerows(tables['time'])

    # Save quality tables
    for dataset in datasets:
        with open(penne.RESULTS_DIR / f'quality-{dataset}.csv', 'w') as file:

            # Setup csv writer
            writer = csv.DictWriter(
                file,
                fieldnames=['name', 'f1', 'l1', 'rca', 'rpa', 'threshold'],
                delimiter=',')

            # Write to csv
            writer.writeheader()
            writer.writerows(tables['quality'][dataset])


def parse_f1(results):
    """Get optimal F1 and corresponding threshold"""
    # Get f1s
    f1s = {
        key: value for key, value in results.items() if key.startswith('f1')}

    # Get optimal key
    key = max(f1s, key=f1s.get)

    # Parse F1 and threshold
    return f1s[key], float(key[3:])


def quality_results(directory, dataset):
    """Update quality results table"""
    # Load results
    with open(directory / 'overall.json') as file:
        results = json.load(file)[dataset]

    # Get optimal f1 and corresponding threshold
    f1, threshold = parse_f1(results)

    # Populate table
    return {
        'f1': f1,
        'l1': results['l1'],
        'name': directory.name,
        'rca': results['rca'],
        'rpa': results['rpa'],
        'threshold': threshold}


def time_results(directory):
    """Update timing results table"""
    # Load results
    with open(directory / 'time.json') as file:
        results = json.load(file)

    # Format table row
    return {
        'name': directory.name,
        'rtf-cpu': results['cpu']['real-time-factor'],
        'rtf-gpu': results['gpu']['real-time-factor']}
