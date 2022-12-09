"""Config parameters whose values depend on other config parameters"""
import penne


###############################################################################
# Audio parameters
###############################################################################


# Maximum representable frequency
FMAX = \
    penne.FMIN * 2 ** (penne.PITCH_BINS * penne.CENTS_PER_BIN / penne.OCTAVE)

# Hopsize in seconds
HOPSIZE_SECONDS = penne.HOPSIZE / penne.SAMPLE_RATE


###############################################################################
# Directories
###############################################################################


# Location to save dataset partitions
PARTITION_DIR = penne.ASSETS_DIR / 'partitions'

# Default checkpoint for generation
DEFAULT_CHECKPOINT = penne.ASSETS_DIR / 'checkpoints' / 'default.pt'


###############################################################################
# Evaluation
###############################################################################


# Timer for benchmarking generation
TIMER = penne.time.Context()


###############################################################################
# Training parameters
###############################################################################


# Number of samples used during training
NUM_TRAINING_SAMPLES = \
    (penne.NUM_TRAINING_FRAMES - 1) * penne.HOPSIZE + penne.WINDOW_SIZE
