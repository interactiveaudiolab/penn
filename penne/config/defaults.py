from pathlib import Path

import numpy as np


###############################################################################
# Metadata
###############################################################################


# Configuration name
CONFIG = 'penne'


###############################################################################
# Audio parameters
###############################################################################


# Width of a pitch bin
CENTS_PER_BIN = 20  # cents

# Whether to trade quantization error for noise during inference
DITHER = False

# Distance between adjacent frames
HOPSIZE = 160  # samples

# Maximum representable frequency
MAX_FMAX = 2006.  # hz

# Number of pitch bins to predict
PITCH_BINS = 360

# Audio sample rate
SAMPLE_RATE = 16000  # hz

# Token indicating no pitch is present
UNVOICED = np.nan

# Size of the analysis window
WINDOW_SIZE = 1024  # samples


###############################################################################
# Directories
###############################################################################


# Location to save assets to be bundled with pip release
ASSETS_DIR = Path(__file__).parent.parent / 'assets'

# Location of preprocessed features
CACHE_DIR = Path(__file__).parent.parent.parent / 'data' / 'cache'

# Location of datasets on disk
DATA_DIR = Path(__file__).parent.parent.parent / 'data' / 'datasets'

# Location to save evaluation artifacts
EVAL_DIR = Path(__file__).parent.parent.parent / 'eval'

# Location to save training and adaptation artifacts
RUNS_DIR = Path(__file__).parent.parent.parent / 'runs'

# Location of compressed datasets on disk
SOURCE_DIR = Path(__file__).parent.parent.parent / 'data' / 'sources'


###############################################################################
# Logging parameters
###############################################################################


# Number of steps between saving checkpoints
CHECKPOINT_INTERVAL = 10000  # steps

# Number of steps between logging to Tensorboard
LOG_INTERVAL = 250  # steps

# TODO
LOG_EXAMPLE = 'PTDB' # plot posterior distribution for example of LOG_EXAMPLE dataset
LOG_EXAMPLE_FREQUENCY = 2 # plot posterior distribution every LOG_EXAMPLE_FREQENCY epochs
LOG_WITH_SOFTMAX = False # true => softmax on posterior distribution logits


###############################################################################
# Training parameters
###############################################################################


# TODO
VOICE_ONLY = False # toggle training with voice only or not

# Batch size (per gpu)
BATCH_SIZE = 64

# TODO
LEARNING_RATE = 2e-4
HARMO_LEARNING_RATE = 1e-3

# Per-epoch decay rate of the learning rate
LEARNING_RATE_DECAY = .999875

# Number of training steps
NUM_STEPS = 300000

# Number of frames used during training
NUM_TRAINING_SAMPLES = WINDOW_SIZE // HOPSIZE

# Number of samples used during training
NUM_TRAINING_SAMPLES = WINDOW_SIZE

# Number of data loading worker threads
NUM_WORKERS = 2

# Seed for all random number generators
RANDOM_SEED = 1234
