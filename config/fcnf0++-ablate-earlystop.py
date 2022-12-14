CONFIG = 'fcnf0++-ablate-earlystop'

# The decoder to use for postprocessing
DECODER = 'argmax'

# Whether to stop training when validation loss stops improving
EARLY_STOPPING = True

# Number of steps between logging to Tensorboard
LOG_INTERVAL = 500  # steps
