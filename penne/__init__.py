# Trainings
# - More bins
# - Crepe original mdb
# - Harmof0 both
# - Harmof0 mdb
# - Harmof0 ptdb
# - No blur
# - Preprocessing?

# Evaluation
# - Dither

# Development
# - user API
# - evaluation timing
# - torchscript
# - periodicity methods


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
from .config.static import *


###############################################################################
# Module imports
###############################################################################


from .core import *
from .model import Model
from . import checkpoint
from . import convert
from . import data
from . import evaluate
from . import load
from . import partition
from . import plot
from . import preprocess
from . import train
from . import write
