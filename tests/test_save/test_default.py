import glob
import json
import os
import pickle
import tempfile
import unittest

import numpy as np

from brain_pipe.save.default import (
    default_metadata_key_fn,
    default_filename_fn,
    DefaultSave,
)
from brain_pipe.utils.serialization import pickle_load_wrapper


class defaultMetadataKeyFnTest(unittest.TestCase):
    def test_defaultMetadataKeyFn(self):
        data1 = {"data_path": "a"}
        self.assertEqual(default_metadata_key_fn(data1), "a")
        data2 = {"stimulus_path": "b"}
        self.assertEqual(default_metadata_key_fn(data2), "b")
        data3 = {"trigger_path": "c"}
        self.assertEqual(default_metadata_key_fn(data3), "c")
        data4 = {"stimulus_path": os.path.join("c", "d"), "trigger_path": "e"}
        self.assertEqual(default_metadata_key_fn(data4), "d")


class defaultFilenameFnTest(unittest.TestCase):
    def test_defaultFilenameFn(self):
        self.assertEqual(
            default_filename_fn({"data_path": "a"}, "b", "c", "_-_"), "c_-_a_-_b.npy"
        )
        self.assertEqual(
            default_filename_fn(
                {"data_path": "a", "stimulus_path": "d"}, None, None, "_"
            ),
            "a_d.data_dict",
        )
        self.assertEqual(
            default_filename_fn(
                {
                    "data_path": "a",
                    "stimulus_path": "d",
                    "event_info": {"snr": 123.123},
                },
                "b",
                "c",
                "|",
            ),
            "c|a|d|123.123|b.npy",
        )


class SaveTest(unittest.TestCase):
    class MockupFilenameFn:
        def __call__(self, data_dict, feature_name, set_name=None, separator="_-_"):
            output = data_dict["output_filename"]
            if set_name is not None:
                output = set_name + separator + output
            return output

    class MockupMetadataKeyFn:
        def __call__(self, data_dict):
            return data_dict["metadata_key"]

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()

    def test_is_already_done(self):
        saver = DefaultSave(
            self.tmp_dir.name,
            filename_fn=self.MockupFilenameFn(),
            metadata_key_fn=self.MockupMetadataKeyFn(),
        )
        path = os.path.join(self.tmp_dir.name, "b")
        temp_dict = {"output_filename": path, "metadata_key": "d"}
        # Not done yet
        self.assertFalse(saver.is_already_done(temp_dict))

        with open(os.path.join(self.tmp_dir.name, ".save_metadata.json"), "w") as f:
            json.dump({"d": os.path.basename(path)}, f)

        with open(path, "w") as f:
            f.write("")
        # Done now
        self.assertTrue(saver.is_already_done(temp_dict))

        # Add a path
        path2 = os.path.join(self.tmp_dir.name, "c")
        with open(os.path.join(self.tmp_dir.name, ".save_metadata.json"), "w") as f:
            json.dump({"d": [path, path2]}, f)
        # Not done anymore
        self.assertFalse(saver.is_already_done(temp_dict))

        # Create the second file
        with open(path2, "w") as f:
            f.write("")
        # Done again
        self.assertTrue(saver.is_already_done(temp_dict))

        # When overwrite is True, it should always return False
        saver = DefaultSave(self.tmp_dir.name, overwrite=True)
        self.assertFalse(saver.is_already_done({}))

    def test_is_reloadable(self):
        saver = DefaultSave(
            self.tmp_dir.name,
            filename_fn=self.MockupFilenameFn(),
            metadata_key_fn=self.MockupMetadataKeyFn(),
        )
        path = os.path.join(self.tmp_dir.name, "b")
        temp_dict = {"output_filename": path, "metadata_key": "d"}
        # Nothing to reload
        self.assertFalse(saver.is_reloadable(temp_dict))

        with open(os.path.join(self.tmp_dir.name, ".save_metadata.json"), "w") as f:
            json.dump({"d": os.path.basename(path)}, f)

        # There is a path to reload, but the file does not exist
        self.assertFalse(saver.is_reloadable(temp_dict))

        with open(path, "w") as f:
            f.write("")
        # There is a path to reload, and the file exists
        self.assertTrue(saver.is_reloadable(temp_dict))

        saver = DefaultSave(
            self.tmp_dir.name,
            filename_fn=self.MockupFilenameFn(),
            metadata_key_fn=self.MockupMetadataKeyFn(),
            overwrite=True,
        )
        # When overwrite is True, it should always return False
        self.assertFalse(saver.is_reloadable(temp_dict))

        # TODO: Change when loading multiple files is implemented
        path2 = os.path.join(self.tmp_dir.name, "c")
        with open(os.path.join(self.tmp_dir.name, ".save_metadata.json"), "w") as f:
            json.dump({"d": [path, path2]}, f)
        with open(path2, "w") as f:
            f.write("")
        self.assertFalse(saver.is_reloadable(temp_dict))

    def test_reload(self):
        saver = DefaultSave(
            self.tmp_dir.name,
            filename_fn=self.MockupFilenameFn(),
            metadata_key_fn=self.MockupMetadataKeyFn(),
        )
        path = os.path.join(self.tmp_dir.name, "b")
        temp_dict = {"output_filename": path, "metadata_key": "d"}
        # Nothing to reload
        with self.assertRaises(KeyError):
            saver.reload(temp_dict)

        with open(os.path.join(self.tmp_dir.name, ".save_metadata.json"), "w") as f:
            json.dump({"d": os.path.basename(path)}, f)

        # There is a path to reload, but the file does not exist
        with self.assertRaises(ValueError):
            saver.reload(temp_dict)

        data_dict = {"data": 123}
        with open(path, "wb") as f:
            pickle.dump(data_dict, f)

        # There is a path to reload, and the file exists, but reload_fn is not set
        # to handle no suffixes
        with self.assertRaises(ValueError):
            saver.reload(temp_dict)

        # Set reload_fn to handle no suffixes
        saver.reload_fn = pickle_load_wrapper
        # Should work now
        self.assertEqual(saver.reload(temp_dict), data_dict)
        # Reset reload_fn
        saver.reload_fn = saver.DEFAULT_RELOAD_FUNCTIONS

        # Try with .data_dict suffix
        path2 = os.path.join(self.tmp_dir.name, "b.data_dict")
        with open(os.path.join(self.tmp_dir.name, ".save_metadata.json"), "w") as f:
            json.dump({"d": os.path.basename(path2)}, f)
        with open(path2, "wb") as f:
            pickle.dump(data_dict, f)

        temp_dict = {"output_filename": path2, "metadata_key": "d"}
        # Should work now
        self.assertEqual(saver.reload(temp_dict), data_dict)

    def test_call(self):
        saver = DefaultSave(
            self.tmp_dir.name,
            filename_fn=self.MockupFilenameFn(),
            metadata_key_fn=self.MockupMetadataKeyFn(),
        )
        data_dict = {"output_filename": "a.pickle", "metadata_key": "b", "data": 123}
        output = saver(data_dict)
        with open(os.path.join(self.tmp_dir.name, ".save_metadata.json"), "r") as f:
            self.assertEqual(json.load(f), {"b": ["a.pickle"]})
        with open(os.path.join(self.tmp_dir.name, "a.pickle"), "rb") as f:
            self.assertEqual(pickle.load(f), data_dict)
        self.assertEqual(output, data_dict)

    def test_to_save(self):
        saver = DefaultSave(
            self.tmp_dir.name,
            filename_fn=self.MockupFilenameFn(),
            metadata_key_fn=self.MockupMetadataKeyFn(),
            to_save={"envelope": "data"},
        )
        data_dict = {"output_filename": "a.npy", "metadata_key": "b", "data": 123}
        output = saver(data_dict)
        with open(os.path.join(self.tmp_dir.name, ".save_metadata.json"), "r") as f:
            self.assertEqual(json.load(f), {"b": ["a.npy"]})
        self.assertEqual(
            np.load(os.path.join(self.tmp_dir.name, "a.npy")), np.array(123)
        )
        self.assertEqual(output, data_dict)

        # Test to_save with multiple set_names
        saver.overwrite = True
        data_dict = {
            "output_filename": "a.npy",
            "metadata_key": "b",
            "data": {"train": 1, "validate": 2, "test": 3},
        }
        output = saver(data_dict)
        self.assertEqual(output, data_dict)
        paths = glob.glob(os.path.join(self.tmp_dir.name, "*_-_a.npy"))
        self.assertEqual(len(paths), 3)
        for path, value in zip(sorted(paths), [3, 1, 2]):
            self.assertEqual(np.load(path), np.array(value))
        with open(os.path.join(self.tmp_dir.name, ".save_metadata.json"), "r") as f:
            self.assertEqual(
                json.load(f),
                {"b": ["train_-_a.npy", "validate_-_a.npy", "test_-_a.npy"]},
            )

    def test_clear_output(self):
        data_dict = {"output_filename": "a.data_dict", "metadata_key": "b", "data": 123}
        # Test clear output
        saver = DefaultSave(
            self.tmp_dir.name,
            filename_fn=self.MockupFilenameFn(),
            metadata_key_fn=self.MockupMetadataKeyFn(),
            clear_output=True,
        )
        self.assertIsNone(saver(data_dict))

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()
