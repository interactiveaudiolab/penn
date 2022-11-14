CONFIG = 'fcnf0'

# Batch size
BATCH_SIZE = 32

# Width of a pitch bin
CENTS_PER_BIN = 25  # cents

# Whether to peak-normalize CREPE input audio
CREPE_NORMALIZE = True

# The dropout rate. Set to None to turn off dropout.
DROPOUT = None

# Minimum representable frequency
FMIN = 30.  # Hz

# Whether to stop training when validation loss stops improving
EARLY_STOPPING = True

# Number of steps between logging to Tensorboard
LOG_INTERVAL = 500  # steps

# Number of batches to use for validation
LOG_STEPS = 4

# The name of the model to use for training
MODEL = 'fcnf0'

# Number of pitch bins to predict
PITCH_BINS = 486

# Whether to only use voiced start frames
VOICED_ONLY = True
