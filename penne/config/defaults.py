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

# Whether to peak-normalize CREPE input audio
CREPE_NORMALIZE = False

# Whether to trade quantization error for noise during inference
DITHER = False

# Minimum representable frequency
FMIN = 31.  # Hz

# Distance between adjacent frames
HOPSIZE = 160  # samples

# Number of spectrogram frequency bins
NUM_FFT = 1024

# One octave in cents
OCTAVE = 1200  # cents

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
# Evaluation parameters
###############################################################################


# Whether to perform benchmarking
BENCHMARK = False

# Number of steps between saving checkpoints
CHECKPOINT_INTERVAL = 25000  # steps

# Batch size to use for evaluation
EVALUATION_BATCH_SIZE = 2048

# Number of steps between logging to Tensorboard
LOG_INTERVAL = 2500  # steps

# Number of batches to use for validation
LOG_STEPS = 64

# Method to use for periodicity extraction
PERIODICITY = 'max'

# Perform inference using torchscript optimization
TORCHSCRIPT = False


###############################################################################
# Model parameters
###############################################################################


# The dropout rate. Set to None to turn off dropout.
DROPOUT = .25

# The max pooling kernel and stride. Set to None to turn off max pooling.
MAX_POOL = (2, 2)

# The name of the model to use for training
MODEL = 'crepe'

# Type of model normalization
NORMALIZATION = 'batch'


###############################################################################
# Training parameters
###############################################################################


# Whether to use adaptive gradient clipping
ADAPTIVE_CLIPPING = False

# Batch size (per gpu)
BATCH_SIZE = 64

# Weight applied to positive examples in binary cross-entropy loss
BCE_POSITIVE_WEIGHT = 1.

# Whether to stop training when validation loss stops improving
EARLY_STOPPING = False

# Whether to apply Gaussian blur to binary cross-entropy loss targets
GAUSSIAN_BLUR = True

# Optimizer learning rate
LEARNING_RATE = 2e-4

# Loss function
LOSS = 'binary_cross_entropy'

# Number of training steps
NUM_STEPS = 250000

# Number of frames used during training
NUM_TRAINING_FRAMES = 1

# Number of data loading worker threads
NUM_WORKERS = 4

# Seed for all random number generators
RANDOM_SEED = 1234

# Whether to only use voiced start frames
VOICED_ONLY = False
