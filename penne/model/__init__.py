from .crepe import Crepe
from .harmof0 import Harmof0

import penne


def Model(name=penne.MODEL):
    """Create a model"""
    if name == 'crepe':
        return Crepe()
    if name == 'harmof0':
        return Harmof0()
    raise ValueError(f'Model {name} is not defined')
