CONFIG = 'crepe-original-ptdb'

# Batch size (per gpu)
BATCH_SIZE = 32

# Whether to peak-normalize CREPE input audio
CREPE_NORMALIZE = True

# Whether to stop training when validation loss stops improving
EARLY_STOPPING = True

# Number of steps between logging to Tensorboard
LOG_INTERVAL = 500  # steps

# Number of batches to use for validation
LOG_STEPS = 1

# Whether to only use voiced start frames
VOICED_ONLY = True
