CONFIG = 'fcnf0'

# Batch size
BATCH_SIZE = 32

# Width of a pitch bin
CENTS_PER_BIN = 25  # cents

# The decoder to use for postprocessing
DECODER = 'argmax'

# Whether to stop training when validation loss stops improving
EARLY_STOPPING = True

# Minimum representable frequency
FMIN = 30.  # Hz

# Number of steps between logging to Tensorboard
LOG_INTERVAL = 500  # steps

# Loss function
LOSS = 'binary_cross_entropy'

# The name of the model to use for training
MODEL = 'fcnf0'

# Whether to peak-normalize CREPE input audio
NORMALIZE_INPUT = True

# Type of model normalization
NORMALIZATION = 'batch'

# Number of pitch bins to predict
PITCH_BINS = 486

# Whether to only use voiced start frames
VOICED_ONLY = True
