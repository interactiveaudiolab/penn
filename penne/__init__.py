# Trainings
# - crepe
# - crepe++
# - deepf0
# - deepf0++
# - fcnf0
# - fcnf0++
# - harmof0++
# - harmof0

# - crepe++-mdb
# - crepe++-ptdb
# - crepe-1440-cce-layer-nodropout-unvoiced-noearly
# - crepe-128-cce-layer-nodropout-unvoiced-noearly
# - crepe-128-1440-layer-nodropout-unvoiced-noearly
# - crepe-128-1440-cce-nodropout-unvoiced-noearly
# - crepe-128-1440-cce-layer-unvoiced-noearly
# - crepe-128-1440-cce-layer-nodropout-unvoiced

# Figures
# - dataset density + true positive density*
# - threshold landscapes

# Development
# - torchcrepe eval**
# - debug harmof0
# - original harmof0

# - torchscript
# - quantization
# - ONNX?

# Paper
# - weighted
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
from . import train
from . import write

# TEMPORARY
from . import temp
