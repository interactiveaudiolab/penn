from pathlib import Path


###############################################################################
# Metadata
###############################################################################


# Configuration name
CONFIG = 'fcnf0++'


###############################################################################
# Audio parameters
###############################################################################


# Width of a pitch bin
CENTS_PER_BIN = 5.  # cents

# Whether to trade quantization error for noise during inference
DITHER = False

# Minimum representable frequency
FMIN = 31.  # Hz

# Distance between adjacent frames
HOPSIZE = 80  # samples

# The size of the window used for locally normal pitch decoding
LOCAL_PITCH_WINDOW_SIZE = 19

# Pitch velocity constraint for viterbi decoding
MAX_OCTAVES_PER_SECOND = 35.92

# Whether to normalize input audio to mean zero and variance one
NORMALIZE_INPUT = False

# Number of spectrogram frequency bins
NUM_FFT = 1024

# One octave in cents
OCTAVE = 1200  # cents

# Number of pitch bins to predict
PITCH_BINS = 1440

# Audio sample rate
SAMPLE_RATE = 8000  # hz

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

# List of all datasets
DATASETS = ['mdb', 'ptdb']

# Method to use for evaluation
METHOD = 'penn'

# Batch size to use for evaluation
EVALUATION_BATCH_SIZE = 2048

# Datsets to use for evaluation
EVALUATION_DATASETS = DATASETS

# Number of steps between logging to Tensorboard
LOG_INTERVAL = 2500  # steps

# Number of batches to use for validation
LOG_STEPS = 64

# Method to use for periodicity extraction
PERIODICITY = 'entropy'


###############################################################################
# Model parameters
###############################################################################


# The decoder to use for postprocessing
DECODER = 'locally_normal'

# The dropout rate. Set to None to turn off dropout.
DROPOUT = None

# The name of the model to use for training
MODEL = 'fcnf0'

# Type of model normalization
NORMALIZATION = 'layer'


###############################################################################
# Training parameters
###############################################################################


# Batch size
BATCH_SIZE = 128

# Default threshold for voiced/unvoiced classification
DEFAULT_VOICING_THRESHOLD = .1625

# Whether to stop training when validation loss stops improving
EARLY_STOPPING = False

# Stop after this number of log intervals without validation improvements
EARLY_STOPPING_STEPS = 32

# Whether to apply Gaussian blur to binary cross-entropy loss targets
GAUSSIAN_BLUR = True

# Optimizer learning rate
LEARNING_RATE = 2e-4

# Loss function
LOSS = 'categorical_cross_entropy'

# Number of training steps
STEPS = 250000

# Number of frames used during training
NUM_TRAINING_FRAMES = 1

# Number of data loading worker threads
NUM_WORKERS = 4

# Seed for all random number generators
RANDOM_SEED = 1234

# Whether to only use voiced start frames
VOICED_ONLY = False
