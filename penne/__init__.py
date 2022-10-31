# Trainings
# - Misalignments

# Evaluation
# - Torchscript
# - Dither
# - torchcrepe
# - original harmof0

# Figures
# - time delay

# Development
# - sgconv**
# - test ddp
# - instance norm**
# - debug harmof0**
# - Local linear weighting**
# - periodicity methods
#    - autocorrelation
#    - cnmdf


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
