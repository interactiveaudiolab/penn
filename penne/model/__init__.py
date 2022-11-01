from .crepe import Crepe
from .deepf0 import Deepf0
from .harmof0 import Harmof0

import penne


def Model(name=penne.MODEL):
    """Create a model"""
    if name == 'crepe':
        return Crepe()
    if name == 'deepf0':
        return Deepf0()
    if name == 'harmof0':
        return Harmof0()
    raise ValueError(f'Model {name} is not defined')
