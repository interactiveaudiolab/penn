CONFIG = 'crepe-original-ptdb'

# Batch size (per gpu)
BATCH_SIZE = 32

# Whether to peak-normalize CREPE input audio
CREPE_NORMALIZE = True

# Whether to stop training when validation loss stops improving
EARLY_STOPPING = True

# Number of steps between logging to Tensorboard
LOG_INTERVAL = 32 * 500  # steps

# Whether to only use voiced start frames
VOICED_ONLY = True
