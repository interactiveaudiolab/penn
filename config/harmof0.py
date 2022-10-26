CONFIG = 'harmof0'

# Batch size (per gpu)
BATCH_SIZE = 24

# Width of a pitch bin
CENTS_PER_BIN = 25  # cents

# Minimum representable frequency
FMIN = 27.5  # Hz

# Whether to apply Gaussian blur to binary cross-entropy loss targets
GAUSSIAN_BLUR = False

# Optimizer learning rate
LEARNING_RATE = 1e-3

# Weight applied to positive examples in binary cross-entropy loss
LOSS_WEIGHT = 20.

# The name of the model to use for training
MODEL = 'harmof0'

# Number of frames used during training
NUM_TRAINING_FRAMES = 200

# Number of samples used during training
NUM_TRAINING_SAMPLES = 32000 + 1024 - 160

# Number of pitch bins to predict
PITCH_BINS = 352
