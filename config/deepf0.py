CONFIG = 'deepf0'

# Batch size
BATCH_SIZE = 32

# Whether to peak-normalize CREPE input audio
CREPE_NORMALIZE = True

# Whether to stop training when validation loss stops improving
EARLY_STOPPING = True

# Number of steps between logging to Tensorboard
LOG_INTERVAL = 500  # steps

# Number of batches to use for validation
LOG_STEPS = 4

# The name of the model to use for training
MODEL = 'deepf0'

# Type of model normalization
NORMALIZATION = 'weight'

# Whether to only use voiced start frames
VOICED_ONLY = True
