"""Config parameters whose values depend on other config parameters"""
import penn


###############################################################################
# Audio parameters
###############################################################################


# Maximum representable frequency
FMAX = \
    penn.FMIN * 2 ** (penn.PITCH_BINS * penn.CENTS_PER_BIN / penn.OCTAVE)

# Hopsize in seconds
HOPSIZE_SECONDS = penn.HOPSIZE / penn.SAMPLE_RATE


###############################################################################
# Directories
###############################################################################


# Location to save dataset partitions
PARTITION_DIR = penn.ASSETS_DIR / 'partitions'

# Default checkpoint for generation
DEFAULT_CHECKPOINT = penn.ASSETS_DIR / 'checkpoints' / 'fcnf0++.pt'


###############################################################################
# Evaluation
###############################################################################


# Timer for benchmarking generation
TIMER = penn.time.Context()


###############################################################################
# Training parameters
###############################################################################


# Number of samples used during training
NUM_TRAINING_SAMPLES = \
    (penn.NUM_TRAINING_FRAMES - 1) * penn.HOPSIZE + penn.WINDOW_SIZE
