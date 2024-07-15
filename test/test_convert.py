import penn

import librosa
import torch


###############################################################################
# Test convert.py
###############################################################################


def test_convert_frequency_to_midi():
    """Test that conversion from Hz to MIDI matches librosa implementation"""
    sample_data = torch.tensor([110.0, 220.0, 440.0, 500.0, 880.0])

    # Convert
    penn_midi = penn.convert.frequency_to_midi(sample_data)
    librosa_midi = librosa.hz_to_midi(sample_data.numpy())

    # Compare
    assert torch.allclose(
        penn_midi,
        torch.tensor(librosa_midi, dtype=torch.float32))


def test_convert_midi_to_frequency():
    """Test that conversion from MIDI to Hz matches librosa implementation"""
    sample_data = torch.tensor([45.0, 57.0, 69.0, 71.2131, 81.0])

    # Convert
    penn_frequency = penn.convert.midi_to_frequency(sample_data)
    librosa_frequency = librosa.midi_to_hz(sample_data.numpy())

    # Compare
    assert torch.allclose(
        penn_frequency,
        torch.tensor(librosa_frequency, dtype=torch.float32))
