CONFIG = 'harmof0-mdb'

# Batch size (per gpu)
BATCH_SIZE = 24

# Weight applied to positive examples in binary cross-entropy loss
BCE_POSITIVE_WEIGHT = 20.

# Width of a pitch bin
CENTS_PER_BIN = 25  # cents

# Minimum representable frequency
FMIN = 27.5  # Hz

# Whether to apply Gaussian blur to binary cross-entropy loss targets
GAUSSIAN_BLUR = False

# Optimizer learning rate
LEARNING_RATE = 1e-3

# The name of the model to use for training
MODEL = 'harmof0'

# Number of frames used during training
NUM_TRAINING_FRAMES = 200

# Number of pitch bins to predict
PITCH_BINS = 352
