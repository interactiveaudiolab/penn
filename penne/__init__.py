# Development
# - sphinx
# - Huggingface models
# - Yapecs passwords
# - Soft version pinning

# Readme
# - API

# Paper
# - TSAP template
# - Pitch posteriorgram figure (title figure; rhapsody in blue)
# - Unvoiced training threshold figure
# - Dither figure
# - Cross-domain
#   - Figure
#   - Text


###############################################################################
# Configuration
###############################################################################


# Default configuration parameters to be modified
from .config import defaults

# Modify configuration
import yapecs
yapecs.configure(defaults)

# Import configuration parameters
from .config.defaults import *
from . import time
from .config.static import *


###############################################################################
# Module imports
###############################################################################


from .core import *
from .model import Model
from . import checkpoint
from . import convert
from . import data
from . import decode
from . import dsp
from . import evaluate
from . import load
from . import onnx
from . import partition
from . import periodicity
from . import plot
from . import train
from . import write
