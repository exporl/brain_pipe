"""Tests for the auditory EEG dataset example."""
import gzip
import json
import os
import pickle
import sys
import tempfile
import time
import unittest

import librosa
import numpy as np

from brain_pipe.cli.default import DefaultCLI
from brain_pipe.preprocessing.stimulus.audio.spectrogram import LibrosaMelSpectrogram
from brain_pipe.utils.serialization import pickle_load_wrapper
from examples.exporl.sparrKULee import (
    default_librosa_load_fn,
    default_npz_load_fn,
    SparrKULeeSpectrogramKwargs,
    run_preprocessing_pipeline,
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


class DefaultNpzLoadFnTest(unittest.TestCase):
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


class SparrKULeeSpectrogramKwargsTest(unittest.TestCase):
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "test_data",
        "OSR_us_000_0010_8k_mel_challenge.npy",
    )

    def test_sparrkulee(self):
        kwargs = SparrKULeeSpectrogramKwargs()
        data, sr = librosa.load(test_wav_path, sr=None)
        output = LibrosaMelSpectrogram(power_factor=0.6, librosa_kwargs=kwargs)(
            {"stimulus_data": data, "stimulus_sr": sr}
        )
        self.assertTrue(
            np.allclose(output["spectrogram_data"], np.load(self.data_path))
        )


class TruncatedSparrKULeeTest(unittest.TestCase):
    dataset_folder = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "test_data",
        "truncated_sparrKULee",
    )

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()

    def test_script(self):
        eeg_folder = os.path.join(self.tmp_dir.name, "eeg")
        stimuli_folder = os.path.join(self.tmp_dir.name, "stimuli")
        start_time = time.time()
        run_preprocessing_pipeline(
            self.dataset_folder,
            stimuli_folder,
            eeg_folder,
            log_path=os.path.join(self.tmp_dir.name, "log.txt"),
        )
        time_0 = time.time() - start_time
        self._check_output()

        # TODO improve this, this is test is based on timing: i.e.
        #   if the data is reloaded, it will take less long than if it is
        #   reprocessed
        start_time = time.time()
        run_preprocessing_pipeline(
            self.dataset_folder,
            stimuli_folder,
            eeg_folder,
            log_path=os.path.join(self.tmp_dir.name, "log.txt"),
        )
        time_1 = time.time() - start_time
        self._check_output()
        self.assertGreater(time_0, time_1)

    def _check_output(self):
        # TODO check if no extra files are created
        # Walk through the derivative folders and check that the files are there
        derivatives_folder = os.path.join(self.dataset_folder, "derivatives")
        for root, dirs, files in os.walk(derivatives_folder):
            for file in files:
                correct_path = os.path.join(root, file)
                rel_path = os.path.relpath(correct_path, derivatives_folder)
                found_path = os.path.join(self.tmp_dir.name, rel_path)
                if found_path.endswith(".npy"):
                    self.assertTrue(
                        np.allclose(np.load(correct_path), np.load(found_path))
                    )
                elif found_path.endswith(".json"):
                    with open(correct_path, "r") as f:
                        correct_json = json.load(f)
                    with open(found_path, "r") as f:
                        found_json = json.load(f)
                    self.assertEqual(correct_json, found_json)
                elif found_path.endswith(".data_dict.gz"):
                    # Compare data_dict files
                    with gzip.open(correct_path, "rb") as f:
                        correct_dict = pickle.load(f)
                    found_dict = pickle_load_wrapper(found_path[:-3])
                    for key in ["envelope_data", "spectrogram_data", "trigger_data"]:
                        self.assertTrue(np.allclose(correct_dict[key], found_dict[key]))
                elif found_path.endswith("log.txt") or found_path.endswith(".log"):
                    continue
                else:
                    # Compare byte streams
                    with open(correct_path, "rb") as f:
                        correct_bytes = f.read()
                    with open(found_path, "rb") as f:
                        found_bytes = f.read()
                    self.assertEqual(correct_bytes, found_bytes)

    def test_yaml(self):
        cli = DefaultCLI()
        this_folder = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(this_folder))),
            "examples",
            "exporl",
            "sparrKULee.yaml",
        )
        start_time = time.time()
        cli.run(
            [
                yaml_path,
                "--dataset_dir",
                self.dataset_folder,
                "--output_dir",
                self.tmp_dir.name,
            ]
        )
        time_0 = time.time() - start_time
        self._check_output()

        # TODO improve this, this is test is based on timing: i.e.
        #   if the data is reloaded, it will take less long than if it is
        #   reprocessed
        start_time = time.time()
        cli.run(
            [
                yaml_path,
                "--dataset_dir",
                self.dataset_folder,
                "--output_dir",
                self.tmp_dir.name,
            ]
        )
        time_1 = time.time() - start_time
        self._check_output()
        self.assertGreater(time_0, time_1)

    def tearDown(self) -> None:
        if self.tmp_dir is not None:
            self.tmp_dir.cleanup()
