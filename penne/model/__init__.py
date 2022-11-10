from .crepe import Crepe
from .deepf0 import Deepf0
from .fcnf0 import Fcnf0
from .harmof0 import Harmof0
from .s4f0 import S4f0

import penne


def Model(name=penne.MODEL):
    """Create a model"""
    if name == 'crepe':
        return Crepe()
    if name == 'deepf0':
        return Deepf0()
    if name == 'fcnf0':
        return Fcnf0()
    if name == 'harmof0':
        return Harmof0()
    if name == 's4f0':
        return S4f0()
    raise ValueError(f'Model {name} is not defined')
