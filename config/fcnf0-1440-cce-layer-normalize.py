CONFIG = 'fcnf0-1440-cce-layer-nodropout-normalize'

# Width of a pitch bin
CENTS_PER_BIN = 5  # cents

# Whether to peak-normalize CREPE input audio
CREPE_NORMALIZE = True

# The dropout rate. Set to None to turn off dropout.
DROPOUT = None

# Loss function
LOSS = 'categorical_cross_entropy'

# The name of the model to use for training
MODEL = 'fcnf0'

# Type of model normalization
NORMALIZATION = 'layer'

# Number of pitch bins to predict
PITCH_BINS = 1440
