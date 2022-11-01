import argparse

import penne


###############################################################################
# Create figure
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Create delay figure')
    # TODO - other args
    parser.add_argument(
        '--gpu',
        type=int,
        help='The index of the GPU to use for inference')


penne.plot.delay.from_file_to_file(**vars(parse_args()))
