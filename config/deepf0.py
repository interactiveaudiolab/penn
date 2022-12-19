MODULE = 'penn'

# Configuration name
CONFIG = 'deepf0'

# Batch size
BATCH_SIZE = 32

# Width of a pitch bin
CENTS_PER_BIN = 20.  # cents

# The decoder to use for postprocessing
DECODER = 'argmax'

# Whether to stop training when validation loss stops improving
EARLY_STOPPING = True

# Distance between adjacent frames
HOPSIZE = 160  # samples

# Number of steps between logging to Tensorboard
LOG_INTERVAL = 500  # steps

# Loss function
LOSS = 'binary_cross_entropy'

# The name of the model to use for training
MODEL = 'deepf0'

# Type of model normalization
NORMALIZATION = 'weight'

# Whether to peak-normalize CREPE input audio
NORMALIZE_INPUT = True

# Number of pitch bins to predict
PITCH_BINS = 360

# Audio sample rate
SAMPLE_RATE = 16000  # hz

# Whether to only use voiced start frames
VOICED_ONLY = True
