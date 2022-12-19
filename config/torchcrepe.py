import torchcrepe

MODULE = 'penn'

# Configuration name
# Note - We're not actually training torchcrepe. We only use this for
#        evaluation, and only use the FMIN in order to precisely align
#        predictions with ground truth pitch bins. The other arguments are
#        for completeness. The public crepe (and torchcrepe) model was trained
#        on a set of six datasets, five of which are not considered in the
#        current project.
CONFIG = 'torchcrepe'

# Batch size
BATCH_SIZE = 32

# Width of a pitch bin
CENTS_PER_BIN = 20.  # cents

# The decoder to use for postprocessing
DECODER = 'argmax'

# The dropout rate. Set to None to turn off dropout.
DROPOUT = .25

# Whether to stop training when validation loss stops improving
EARLY_STOPPING = True

# Exactly align pitch bins
FMIN = torchcrepe.convert.cents_to_frequency(1997.3794084376191)

# Distance between adjacent frames
HOPSIZE = 160  # samples

# Number of steps between logging to Tensorboard
LOG_INTERVAL = 500  # steps

# Loss function
LOSS = 'binary_cross_entropy'

# The pitch estimation method to use
METHOD = 'torchcrepe'

# The name of the model to use for training
MODEL = 'crepe'

# Type of model normalization
NORMALIZATION = 'batch'

# Whether to peak-normalize CREPE input audio
NORMALIZE_INPUT = True

# Number of pitch bins to predict
PITCH_BINS = 360

# Audio sample rate
SAMPLE_RATE = 16000  # hz

# Whether to only use voiced start frames
VOICED_ONLY = True
