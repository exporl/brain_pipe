"""Tests for the auditory EEG dataset example."""
import os
import tempfile
import unittest

import numpy as np

from examples.exporl.sparrKULee import (
    default_librosa_load_fn,
    default_npz_load_fn,
)

test_data_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "test_data",
)
test_wav_path = os.path.join(test_data_path, "OSR_us_000_0010_8k.wav")


class DefaultLibrosaLoadFnTest(unittest.TestCase):
    def test_defaultLibrosaLoadFn(self):
        data = default_librosa_load_fn(test_wav_path)
        self.assertEqual(data["sr"], 8000)
        self.assertEqual(data["data"].shape, (268985,))


class defaultNpzLoadFnTest(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(42)
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.data = np.random.normal(0, 1, size=(1000,))
        self.fs = 50
        np.savez(self.temp_file, audio=self.data, fs=self.fs)
        self.temp_file.seek(0)

    def test_defaultNpzLoadFn(self):
        data = default_npz_load_fn(self.temp_file.name)
        self.assertEqual(data["sr"], self.fs)
        self.assertEqual(data["data"].shape, self.data.shape)
        self.assertTrue(np.allclose(data["data"], self.data))

    def tearDown(self) -> None:
        if os.path.exists(self.temp_file.name):
            os.remove(self.temp_file.name)
