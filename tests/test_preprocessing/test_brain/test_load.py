import os
import unittest

from brain_pipe.preprocessing.brain.load import MNELoader


class MNELoaderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test_folder = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        self.test_data_folder = os.path.join(self.test_folder, "test_data")
        self.bdf1_path = os.path.join(self.test_data_folder, "Newtest17-256.bdf")
        self.bdf2_path = os.path.join(self.test_data_folder, "Newtest17-2048.bdf.gz")

    def test_load(self):
        loader = MNELoader()
        output = loader({"data_path": self.bdf1_path})
        self.assertEqual(len(output["raw"].ch_names), 17)
        self.assertEqual(output["raw"].n_times, 15360)

    def test_load_gzip(self):
        loader = MNELoader()
        output = loader({"data_path": self.bdf2_path})
        self.assertEqual(len(output["raw"].ch_names), 17)
        self.assertEqual(output["raw"].n_times, 122880)

    def test_load_multiple(self):
        loader = MNELoader({"data_path": "raw", "data_path2": "raw2"})
        output = loader({"data_path": self.bdf1_path, "data_path2": self.bdf2_path})
        self.assertEqual(len(output["raw"].ch_names), 17)
        self.assertEqual(output["raw"].n_times, 15360)
        self.assertEqual(len(output["raw2"].ch_names), 17)
        self.assertEqual(output["raw2"].n_times, 122880)

    def test_load_extra_args(self):
        loader = MNELoader(include="Status")
        output = loader({"data_path": self.bdf1_path})
        self.assertEqual(output["raw"].ch_names, ["Status"])
