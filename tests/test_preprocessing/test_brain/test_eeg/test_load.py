import os
import unittest

import numpy as np

from brain_pipe.preprocessing.brain.eeg.load import LoadEEGNumpy


class LoadEEGNumpyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test_folder = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        self.bdf_path = os.path.join(self.test_folder, "test_data", "Newtest17-256.bdf")

    def test_get_channels(self):
        loader = LoadEEGNumpy()
        data = np.reshape(np.arange(16), (4, 4))

        # Select by name
        output = loader.get_channels(data, ["a", "b", "c", "d"], ["b"])
        self.assertTrue((output == [[4, 5, 6, 7]]).all())

        # Select by index
        output = loader.get_channels(data, ["a", "b", "c", "d"], [0])
        self.assertTrue((output == [[0, 1, 2, 3]]).all())

        # Select everything
        output = loader.get_channels(data, ["a", "b", "c", "d"], None)
        self.assertTrue((output == data).all())

        # Invalid name
        with self.assertRaises(KeyError):
            loader.get_channels(data, ["a", "b", "c", "d"], ["e"])

        # Invalid index
        with self.assertRaises(IndexError):
            loader.get_channels(data, ["a", "b", "c", "d"], [9])

    def test_load(self):
        loader = LoadEEGNumpy()

        output = loader({"data_path": self.bdf_path})
        self.assertEqual(output["data"].shape, (17, 15360))
        self.assertEqual(output["trigger_data"].shape, (1, 15360))
        self.assertEqual(output["eeg_sfreq"], 256.0)

    def test_load_unit_multiplier(self):
        loader = LoadEEGNumpy(unit_multiplier=1.0)
        output = loader({"data_path": self.bdf_path})
        # Check the unit multiplier
        self.assertEqual(output["data"].mean(), 79.21681857950416)

        loader = LoadEEGNumpy(unit_multiplier=0.0)
        output = loader({"data_path": self.bdf_path})
        # Check the unit multiplier
        self.assertEqual(output["data"].sum(), 0)

    def test_load_get_channels(self):
        loader = LoadEEGNumpy(channels_to_select=["A1"])
        output = loader({"data_path": self.bdf_path})
        self.assertEqual(output["data"].shape, (1, 15360))
        self.assertEqual(output["data"][0, 0], -0.0005266094063883666)

        loader = LoadEEGNumpy(channels_to_select=[0, 1])
        output = loader({"data_path": self.bdf_path})
        self.assertEqual(output["data"].shape, (2, 15360))
        self.assertEqual(output["data"][0, 0], -0.0005266094063883666)

        loader = LoadEEGNumpy(channels_to_select=None)
        output = loader({"data_path": self.bdf_path})
        self.assertEqual(output["data"].shape, (17, 15360))

    def test_additional_mapping(self):
        loader = LoadEEGNumpy(additional_mapping={"eeg_sfreq": "test123123"})
        output = loader({"data_path": self.bdf_path})
        self.assertEqual(output["eeg_sfreq"], output["test123123"])
        self.assertEqual(output["eeg_sfreq"], 256)
