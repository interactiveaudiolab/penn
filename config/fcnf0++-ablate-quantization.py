MODULE = 'penn'

# Configuration name
CONFIG = 'fcnf0++-ablate-quantization'

# Width of a pitch bin
CENTS_PER_BIN = 12.5  # cents

# The decoder to use for postprocessing
DECODER = 'argmax'

# Whether to perform local expected value decoding of pitch
LOCAL_EXPECTED_VALUE = False

# Number of pitch bins to predict
PITCH_BINS = 486
