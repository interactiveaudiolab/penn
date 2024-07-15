MODULE = 'penn'

# Configuration name
CONFIG = 'fcnf0++-ablate-chunkviterbi'

# The decoder to use for postprocessing
DECODER = 'viterbi'

# Whether to perform local expected value decoding of pitch
LOCAL_EXPECTED_VALUE = False

# Maximum chunk size for chunked Viterbi decoding
VITERBI_MIN_CHUNK_SIZE = 8
