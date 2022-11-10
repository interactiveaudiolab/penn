import json

import penne


def analyze(runs=None, output_directory=penne.RESULTS_DIR):
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
    tables = {
        'time': {
            'cpu': {},
            'gpu': {}
        },
        'quality': {
            dataset: {
                'f1': {},
                'l1': {},
                'rca': {},
                'rpa': {},
                'threshold': {}
            }
        for dataset in penne.EVALUATION_DATASETS}
    }

    # Populate tables
    for directory in directories:

        # Add timing results
        update_time(tables['time'], directory)

        # Add quality results
        update_quality(tables['quality'], directory)

    # TODO - Save results
    pass


def update_time(table, directory):
    """Update timing results table"""
    # Load results
    with open(directory / '.json') as file:
        results = json.load(file)

    # Populate table
    table[directory.name]['cpu'] = results['cpu']['real-time-factor']
    table[directory.name]['gpu'] = results['gpu']['real-time-factor']


def update_quality(table, directory):
    """Update quality results table"""
    # Load results
    with open(directory / '.json') as file:
        results = json.load(file)

    # Populate table
    for dataset, result in results:
        table[directory.name][dataset]['l1'] = result['l1']
        table[directory.name][dataset]['rca'] = result['rca']
        table[directory.name][dataset]['rpa'] = result['rpa']

        # TODO - f1 and optimal threshold
        pass


