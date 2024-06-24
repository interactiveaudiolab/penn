import penn


###############################################################################
# Test core.py
###############################################################################


def test_infer(audio):
    """Test that inference produces the correct shape"""
    pitch, periodicity = penn.from_audio(
        audio,
        penn.SAMPLE_RATE,
        center='half-hop')
    shape = (1, audio.shape[1] // penn.HOPSIZE)
    assert pitch.shape == periodicity.shape == shape

def test_infer_stereo(audio_stereo):
    """Test that inference on stereo audio produces the correct shape"""
    pitch, periodicity = penn.from_audio(
        audio_stereo,
        penn.SAMPLE_RATE,
        center='half-hop')
    shape = (1, audio_stereo.shape[1] // penn.HOPSIZE)
    assert pitch.shape == periodicity.shape == shape