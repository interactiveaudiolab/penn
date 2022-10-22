"""Config parameters whose values depend on other config parameters"""
import penne


###############################################################################
# Directories
###############################################################################


# Location to save dataset partitions
PARTITION_DIR = penne.ASSETS_DIR / 'partitions'

# Default checkpoint for generation
DEFAULT_CHECKPOINT = penne.ASSETS_DIR / 'checkpoints' / 'default.pt'

# Default configuration file
DEFAULT_CONFIGURATION = penne.ASSETS_DIR / 'configs' / 'default.py'
