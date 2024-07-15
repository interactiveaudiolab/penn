from pathlib import Path

import pytest

import penn


###############################################################################
# Testing fixtures
###############################################################################


@pytest.fixture(scope='session')
def audio():
    """Retrieve the test audio"""
    return penn.load.audio(Path(__file__).parent / 'assets' / 'gershwin.wav')


@pytest.fixture(scope='session')
def audio_stereo():
    """Retrieve the test audio"""
    return penn.load.audio(Path(__file__).parent / 'assets' / '500Hz_stereo.wav')
