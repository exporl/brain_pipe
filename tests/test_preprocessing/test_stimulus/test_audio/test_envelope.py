import os
import unittest

import librosa
import numpy as np
from brian2 import Hz
from brian2hears import Sound

from brain_pipe.preprocessing.stimulus.audio.envelope import (
    EnvelopeFromGammatone,
    GammatoneEnvelope,
)

test_data_path = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
    "test_data",
)
test_wav_path = os.path.join(test_data_path, "OSR_us_000_0010_8k.wav")


class EnvelopeFromGammatoneTest(unittest.TestCase):
    def test_buffer_apply(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, size=(1000,))
        env = EnvelopeFromGammatone(Sound(data, samplerate=10 * Hz), power_factor=0.6)
        self.assertEqual(env.buffer_apply([[0.5]]), np.array([0.5**0.6]))
        self.assertTrue(
            np.isclose(env.buffer_apply([[0.5, -0.5]]), np.array([[1.31950791]]))
        )
        env = EnvelopeFromGammatone(Sound(data, samplerate=10 * Hz), power_factor=0.2)
        self.assertEqual(env.buffer_apply([[0.5]]), np.array([0.5**0.2]))


class GammatoneEnvelopeTest(unittest.TestCase):
    def test_gammatone_envelope(self):
        np.random.seed(42)
        np.random.uniform(-1.0, 1.0, size=(100,))
        data, sr = librosa.load(test_wav_path, sr=None)
        output = GammatoneEnvelope()({"stimulus_data": data, "stimulus_sr": sr})
        self.assertEqual(output["envelope_data"].shape[0], data.shape[0])
