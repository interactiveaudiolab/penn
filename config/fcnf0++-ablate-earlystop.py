MODULE = 'penn'

# Configuration name
CONFIG = 'fcnf0++-ablate-earlystop'

# The decoder to use for postprocessing
DECODER = 'argmax'

# Whether to stop training when validation loss stops improving
EARLY_STOPPING = True

# Whether to perform local expected value decoding of pitch
LOCAL_EXPECTED_VALUE = False

# Number of steps between logging to Tensorboard
LOG_INTERVAL = 500  # steps
