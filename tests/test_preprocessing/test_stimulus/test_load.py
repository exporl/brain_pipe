import os
import unittest

from brain_pipe.preprocessing.stimulus.load import (
    LoadStimuli,
)

test_data_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "test_data",
)
test_wav_path = os.path.join(test_data_path, "OSR_us_000_0010_8k.wav")


class LoadStimuliTest(unittest.TestCase):
    def test_missing_key(self):
        load_stimuli = LoadStimuli()
        # Missing stimulus_path and trigger_path key
        with self.assertRaises(ValueError):
            load_stimuli({})

        # Both keys present but None
        output = load_stimuli({"stimulus_path": None, "trigger_path": None})
        self.assertEqual(output, {"stimulus_path": None, "trigger_path": None})

    def test_load_fn(self):
        load_stimuli = LoadStimuli(load_fn={".wav": lambda x: {"a": 1}})
        output = load_stimuli({"stimulus_path": test_wav_path, "trigger_path": None})
        self.assertEqual(
            output,
            {"stimulus_path": test_wav_path, "trigger_path": None, "stimulus_a": 1},
        )

        load_stimuli = LoadStimuli(load_fn={".nonexistant": lambda x: {}})
        # No load function for .wav
        with self.assertRaises(ValueError):
            load_stimuli({"stimulus_path": test_wav_path, "trigger_path": None})

        load_stimuli = LoadStimuli(load_fn=lambda x: {"a": 1})
        output = load_stimuli({"stimulus_path": test_wav_path, "trigger_path": None})
        self.assertEqual(
            output,
            {"stimulus_path": test_wav_path, "trigger_path": None, "stimulus_a": 1},
        )

    def test_separator(self):
        load_stimuli = LoadStimuli(separator="_-_-_", load_fn=lambda x: {"a": 1})
        output = load_stimuli({"stimulus_path": test_wav_path, "trigger_path": None})
        self.assertEqual(
            output,
            {"stimulus_path": test_wav_path, "trigger_path": None, "stimulus_-_-_a": 1},
        )
