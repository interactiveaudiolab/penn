import json
import math

import penn


###############################################################################
# Create figure
###############################################################################


def from_evaluations(names, evaluations, output_file):
    """Plot periodicity thresholds"""
    import matplotlib.pyplot as plt

    # Create plot
    figure, axis = plt.subplots(figsize=(7, 3))

    # Make pretty
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.spines['bottom'].set_visible(False)
    axis.spines['left'].set_visible(False)
    ticks = [0., .25, .5, .75, 1.]
    axis.set_xlim([0., 1.])
    axis.get_xaxis().set_ticks(ticks)
    axis.get_yaxis().set_ticks(ticks)
    axis.tick_params(axis=u'both', which=u'both',length=0)
    axis.set_xlabel('Unvoiced threshold')
    axis.set_ylabel('F1')
    for tick in ticks:
        axis.axhline(tick, color='gray', linestyle='--', linewidth=.8)

    # Iterate over evaluations to plot
    for name, evaluation in zip(names, evaluations):
        directory = penn.EVAL_DIR / evaluation

        # Load results
        with open(directory / 'overall.json') as file:
            results = json.load(file)['aggregate']
        with open(directory / 'periodicity.json') as file:
            optimal = json.load(file)['entropy']

        # Get thresholds and corresponding F1 values
        x, y = zip(*
            [(key, val) for key, val in results.items() if key.startswith('f1')])
        x = [float(item[3:]) for item in x] + [1]
        y = [0 if math.isnan(item) else item for item in y] + [0]

        # Plot
        line = axis.plot(x, y, label=name)
        color = line[0].get_color()
        axis.plot(optimal['threshold'], optimal['f1'], marker='*', color=color)

    # Add legend
    axis.legend(frameon=False, loc='upper right')

    # Save
    figure.savefig(output_file, bbox_inches='tight', pad_inches=0, dpi=300)
