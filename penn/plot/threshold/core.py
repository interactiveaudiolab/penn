import json
import math

import matplotlib.pyplot as plt

import penn


###############################################################################
# Create figure
###############################################################################


def from_evaluations(evaluations, output_file):
    """Plot periodicity thresholds"""
    # Create plot
    figure, axis = plt.subplots()

    # Make pretty
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.spines['bottom'].set_visible(False)
    axis.spines['left'].set_visible(False)
    # axis.get_xaxis().set_ticks([])
    # axis.get_yaxis().set_ticks([])
    axis.set_xlabel('Unvoiced threshold')
    axis.set_ylabel('F1')

    # Iterate over evaluations to plot
    for evaluation in evaluations:

        # Get evaluation file
        file = penn.EVAL_DIR / evaluation / 'overall.json'

        # Load results
        with open(file) as file:
            results = json.load(file)['aggregate']

        # Get thresholds and corresponding F1 values
        x, y = zip(*
            [(key, val) for key, val in results.items() if key.startswith('f1')])
        x = [float(item[3:]) for item in x]
        y = [0 if math.isnan(item) else item for item in y]

        # Plot
        axis.plot([0] + x, [0] + list(y), label=evaluation)

    # Save
    figure.savefig(output_file, bbox_inches='tight', pad_inches=0, dpi=300)
