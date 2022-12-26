import penn


###############################################################################
# Test core.py
###############################################################################


def test_infer(audio):
    """Test that inference produces the correct shape"""
    pitch, periodicity = penn.from_audio(audio, penn.SAMPLE_RATE, pad=True)
    shape = (1, audio.shape[1] // penn.HOPSIZE)
    assert pitch.shape == periodicity.shape == shape
