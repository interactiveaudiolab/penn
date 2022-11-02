# Trainings
# - ablations
# - Misalignments
# - deepf0 misalign

# Evaluation
# - average + entropy periodicity in automated eval**
# - pyin
# - original harmof0
# - torchcrepe
# - torchscript

# Figures
# - delay
# - logits
# - dataset density + true positive density

# Development
# - debug harmof0**
# - quantization**
# - s4**
# - periodicity methods
#    - autocorrelation
#    - cnmdf

# Paper
# - dither
# - voiced
# - results should not be taken as judgments of speech vs music data being
#   easier or harder

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
