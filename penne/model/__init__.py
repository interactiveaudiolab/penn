from .crepe import Crepe
from .harmof0 import Harmof0


def Model(name):
    """Create a model"""
    if name == 'crepe':
        return Crepe()
    if name == 'harmof0':
        return HarmoF0()
    raise ValueError(f'Model {model} is not defined')
