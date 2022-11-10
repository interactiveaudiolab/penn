# Trainings
# - fcnf0-1440-cce-layer-nodropout-normalize**
# - ablations

# Evaluation
# - analysis module**
# - pyin**
# - fcnf0 eval issues
# - average + entropy periodicity in automated eval**
# - binary entropy search
# - original harmof0
# - torchcrepe
# - torchscript

# Figures
# - dataset density + true positive density
# - threshold landscapes

# Development
# - pyin gpu
# - valid convolutions**
# - quantization**
# - ONNX?
# - debug harmof0**

# Paper
# - weighted**
# - results should not be taken as judgments of speech vs music data being
#   easier or harder
# - unvoiced training causes the landscape of viable thresholds to expand


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
from . import partition
from . import periodicity
from . import plot
from . import preprocess
from . import train
from . import write

# TEMPORARY
from . import temp
