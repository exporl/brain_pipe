import glob
import json
import os
import pickle
import tempfile
import unittest

import numpy as np

from brain_pipe.save.default import (
    default_metadata_key_fn,
    DefaultFilenameFn,
    DefaultSave,
    DefaultSaveMetadata,
)
from brain_pipe.utils.serialization import pickle_load_wrapper


class DefaultMetadataKeyFnTest(unittest.TestCase):
    def test_defaultMetadataKeyFn(self):
        data1 = {"data_path": "a"}
        self.assertEqual(default_metadata_key_fn(data1), "a")
        data2 = {"stimulus_path": "b"}
        self.assertEqual(default_metadata_key_fn(data2), "b")
        data3 = {"trigger_path": "c"}
        self.assertEqual(default_metadata_key_fn(data3), "c")
        data4 = {"stimulus_path": os.path.join("c", "d"), "trigger_path": "e"}
        self.assertEqual(default_metadata_key_fn(data4), "d")


class DefaultFilenameFnTest(unittest.TestCase):
    def test_defaultFilenameFn(self):
        self.assertEqual(
            DefaultFilenameFn(path_keys=["a", "b", "c"])(
                {"a": "c", "b": "a"}, "b", "d"
            ),
            "d_-_c_-_a_-_b.npy",
        )
        self.assertEqual(
            DefaultFilenameFn(separator="_")({"data_path": "a", "stimulus_path": "d"}),
            "a_d.data_dict",
        )
        self.assertEqual(
            DefaultFilenameFn(separator="|")(
                {
                    "data_path": "a",
                    "stimulus_path": "d",
                    "event_info": {"snr": 123.123},
                },
                "b",
                "c",
            ),
            "c|a|d|123.123|b.npy",
        )


class SaveTest(unittest.TestCase):
    class MockupFilenameFn:
        def __init__(self, output_override=None):
            self.output_override = output_override

        def __call__(self, data_dict, feature_name, set_name=None, separator="_-_"):
            if self.output_override is not None:
                output = self.output_override
            else:
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
        metadata_path = os.path.join(self.tmp_dir.name, ".save_metadata.json")
        saver = DefaultSave(
            self.tmp_dir.name,
            filename_fn=self.MockupFilenameFn(),
            metadata=DefaultSaveMetadata(key_fn=self.MockupMetadataKeyFn()),
        )
        path = os.path.join(self.tmp_dir.name, "b")
        temp_dict = {"output_filename": path, "metadata_key": "d"}
        # Not done yet
        self.assertFalse(saver.is_already_done(temp_dict))

        # Make manually sure the file is already done/there.
        with open(metadata_path, "w") as f:
            json.dump({"d": os.path.basename(path)}, f)

        with open(path, "w") as f:
            f.write("")
        # Done now
        self.assertTrue(saver.is_already_done(temp_dict))

        # Add a path that does not exist yet
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
            metadata=DefaultSaveMetadata(key_fn=self.MockupMetadataKeyFn()),
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
            metadata=DefaultSaveMetadata(key_fn=self.MockupMetadataKeyFn()),
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
            metadata=DefaultSaveMetadata(key_fn=self.MockupMetadataKeyFn()),
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
            metadata=DefaultSaveMetadata(key_fn=self.MockupMetadataKeyFn()),
        )
        data_dict = {"output_filename": "a.pickle", "metadata_key": "b", "data": 123}
        output = saver(data_dict)
        with open(os.path.join(self.tmp_dir.name, ".save_metadata.json"), "r") as f:
            self.assertEqual(
                json.load(f),
                {
                    "b": [
                        {"feature_name": None, "filename": "a.pickle", "set_name": None}
                    ]
                },
            )
        with open(os.path.join(self.tmp_dir.name, "a.pickle"), "rb") as f:
            self.assertEqual(pickle.load(f), data_dict)
        self.assertEqual(output, data_dict)

    def test_to_save(self):
        saver = DefaultSave(
            self.tmp_dir.name,
            filename_fn=self.MockupFilenameFn(),
            metadata=DefaultSaveMetadata(key_fn=self.MockupMetadataKeyFn()),
            to_save={"envelope": "data"},
        )
        data_dict = {"output_filename": "a.npy", "metadata_key": "b", "data": 123}
        output = saver(data_dict)
        with open(os.path.join(self.tmp_dir.name, ".save_metadata.json"), "r") as f:
            self.assertEqual(
                json.load(f),
                {
                    "b": [
                        {
                            "feature_name": "envelope",
                            "filename": "a.npy",
                            "set_name": None,
                        }
                    ]
                },
            )
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
                {
                    "b": [
                        {
                            "feature_name": "envelope",
                            "filename": "train_-_a.npy",
                            "set_name": "train",
                        },
                        {
                            "feature_name": "envelope",
                            "filename": "train_-_a.npy",
                            "set_name": "validate",
                        },
                        {
                            "feature_name": "envelope",
                            "filename": "validate_-_a.npy",
                            "set_name": "validate",
                        },
                        {
                            "feature_name": "envelope",
                            "filename": "train_-_a.npy",
                            "set_name": "test",
                        },
                        {
                            "feature_name": "envelope",
                            "filename": "validate_-_a.npy",
                            "set_name": "test",
                        },
                        {
                            "feature_name": "envelope",
                            "filename": "test_-_a.npy",
                            "set_name": "test",
                        },
                    ]
                },
            )

    def test_clear_output(self):
        data_dict = {"output_filename": "a.data_dict", "metadata_key": "b", "data": 123}
        # Test clear output
        saver = DefaultSave(
            self.tmp_dir.name,
            filename_fn=self.MockupFilenameFn(),
            metadata=DefaultSaveMetadata(key_fn=self.MockupMetadataKeyFn()),
            clear_output=True,
        )
        self.assertEqual(saver(data_dict), {})

    def test_multiple_savers(self):
        saver1 = DefaultSave(
            self.tmp_dir.name,
            filename_fn=self.MockupFilenameFn(),
            metadata=DefaultSaveMetadata(key_fn=self.MockupMetadataKeyFn()),
        )
        saver2 = DefaultSave(
            self.tmp_dir.name,
            to_save={"envelope": "data"},
            filename_fn=self.MockupFilenameFn(output_override="a.npy"),
            metadata=DefaultSaveMetadata(key_fn=self.MockupMetadataKeyFn()),
        )

        # Do check_done, check_reloadable and metadata have the correct savers attached?
        self.assertEqual(saver1.check_done.saver, saver1)
        self.assertEqual(saver2.check_done.saver, saver2)
        self.assertEqual(saver1.check_reloadable.saver, saver1)
        self.assertEqual(saver2.check_reloadable.saver, saver2)
        self.assertEqual(saver1.metadata.saver, saver1)
        self.assertEqual(saver2.metadata.saver, saver2)

        data_dict = {
            "output_filename": "a.data_dict",
            "metadata_key": "b",
            "envelope": [[2, 2], [3, 3]],
            "data": 123,
        }

        # Save data
        saver1(data_dict)
        saver2(data_dict)

        # Check the is_already_done
        self.assertTrue(saver1.is_already_done(data_dict))
        self.assertTrue(saver2.is_already_done(data_dict))

        # Only saver1 should be able to reload
        self.assertTrue(saver1.is_reloadable(data_dict))
        self.assertFalse(saver2.is_reloadable(data_dict))

        # Check the reload
        a = saver1.reload(data_dict)
        self.assertEqual(a, data_dict)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()
