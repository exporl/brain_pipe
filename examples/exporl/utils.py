import gzip
import logging
import os
from typing import Dict, Any

import librosa
import numpy as np
import scipy

from brain_pipe.preprocessing.brain.link import BIDSStimulusInfoExtractor


class BIDSAPRStimulusInfoExtractor(BIDSStimulusInfoExtractor):
    """Extract BIDS compliant stimulus information from an .apr file."""

    def __call__(self, brain_dict: Dict[str, Any]):
        """Extract BIDS compliant stimulus information from an events.tsv file.

        Parameters
        ----------
        brain_dict: Dict[str, Any]
            The data dict containing the brain data path.

        Returns
        -------
        Sequence[Dict[str, Any]]
            The extracted event information. Each dict contains the information
            of one row in the events.tsv file
        """
        event_info = super().__call__(brain_dict)
        # Find the apr file
        path = brain_dict[self.brain_path_key]
        apr_path = "_".join(path.split("_")[:-1]) + "_eeg.apr"
        # Read apr file
        apr_data = self.get_apr_data(apr_path)
        # Add apr data to event info
        for e_i in event_info:
            e_i.update(apr_data)
        return event_info

    def get_apr_data(self, apr_path: str):
        """Get the SNR from an .apr file.

        Parameters
        ----------
        apr_path: str
            Path to the .apr file.

        Returns
        -------
        Dict[str, Any]
            The SNR.
        """
        import xml.etree.ElementTree as ET

        apr_data = {}
        tree = ET.parse(apr_path)
        root = tree.getroot()

        # Get SNR
        interactive_elements = root.findall(".//interactive/entry")
        for element in interactive_elements:
            description_element = element.find("description")
            if description_element.text == "SNR":
                apr_data["snr"] = element.find("new_value").text
        if "snr" not in apr_data:
            logging.warning(f"Could not find SNR in {apr_path}.")
            apr_data["snr"] = 100.0
        return apr_data


def default_librosa_load_fn(path):
    """Load a stimulus using librosa.

    Parameters
    ----------
    path: str
        Path to the audio file.

    Returns
    -------
    Dict[str, Any]
        The data and the sampling rate.
    """
    data, sr = librosa.load(path, sr=None)
    return {"data": data, "sr": sr}


def default_npz_load_fn(path):
    """Load a stimulus from a .npz file.

    Parameters
    ----------
    path: str
        Path to the .npz file.

    Returns
    -------
    Dict[str, Any]
        The data and the sampling rate.
    """
    np_data = np.load(path)
    return {
        "data": np_data["audio"],
        "sr": np_data["fs"],
    }


DEFAULT_LOAD_FNS = {
    ".wav": default_librosa_load_fn,
    ".mp3": default_librosa_load_fn,
    ".npz": default_npz_load_fn,
}


def temp_stimulus_load_fn(path):
    """Load stimuli from (Gzipped) files.

    Parameters
    ----------
    path: str
        Path to the stimulus file.

    Returns
    -------
    Dict[str, Any]
        Dict containing the data under the key "data" and the sampling rate
        under the key "sr".
    """
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f_in:
            data = dict(np.load(f_in))
        return {
            "data": data["audio"],
            "sr": data["fs"],
        }

    extension = "." + ".".join(path.split(".")[1:])
    if extension not in DEFAULT_LOAD_FNS:
        raise ValueError(
            f"Can't find a load function for extension {extension}. "
            f"Available extensions are {str(list(DEFAULT_LOAD_FNS.keys()))}."
        )
    load_fn = DEFAULT_LOAD_FNS[extension]
    return load_fn(path)


def bids_filename_fn(data_dict, feature_name, set_name=None):
    """Default function to generate a filename for the data.

    Parameters
    ----------
    data_dict: Dict[str, Any]
        The data dict containing the data to save.
    feature_name: str
        The name of the feature.
    set_name: Optional[str]
        The name of the set. If no set name is given, the set name is not
        included in the filename.

    Returns
    -------
    str
        The filename.
    """

    filename = os.path.basename(data_dict["data_path"]).split("_eeg")[0]

    subject = filename.split("_")[0]
    session = filename.split("_")[1]
    filename += f"_desc-preproc-audio-{os.path.basename(data_dict.get('stimulus_path', '*.')).split('.')[0]}_{feature_name}"

    if set_name is not None:
        filename += f"_set-{set_name}"

    return os.path.join(subject, session, filename + ".npy")


class SparrKULeeSpectrogramKwargs:
    """Default function to generate the kwargs for the librosa spectrogram."""

    def __init__(
        self,
        stimulus_sr_key="stimulus_sr",
        target_fs=64,
        hop_length=None,
        win_length_sec=0.025,
        n_fft=None,
        window_fn=None,
        n_mels=28,
        fmin=-4.2735,
        fmax=5444,
        power=1.0,
        center=False,
        norm=None,
        htk=True,
    ):
        self.stimulus_sr_key = stimulus_sr_key
        self.target_fs = target_fs
        self.hop_length = hop_length
        self.win_length_sec = win_length_sec
        self.n_fft = n_fft
        self.window_fn = window_fn
        if window_fn is None:
            self.window_fn = scipy.signal.windows.hamming
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.power = power
        self.center = center
        self.norm = norm
        self.htk = htk

    def __call__(self, data_dict):
        """Default function to generate the kwargs for the librosa spectrogram.

        Parameters
        ----------
        data_dict: Dict[str, Any]
            The data dict containing the data to save.

        Returns
        -------
        Dict[str, Any]
            The kwargs for the librosa spectrogram.

        Notes
        -----
        Code was based on the code for the 2023 Auditory EEG Challenge code:
        https://github.com/exporl/auditory-eeg-challenge-2023-code/blob/main/
        task1_match_mismatch/util/mel_spectrogram.py
        """
        fs = data_dict[self.stimulus_sr_key]
        result = {
            "fmin": self.fmin,
            "fmax": self.fmax,
            "n_mels": self.n_mels,
            "power": self.power,
            "center": self.center,
            "norm": self.norm,
            "htk": self.htk,
        }

        result["hop_length"] = self.hop_length
        if self.hop_length is None:
            result["hop_length"] = int((1 / self.target_fs) * fs)

        result["win_length"] = self.win_length_sec
        if self.win_length_sec is not None:
            result["win_length"] = int(self.win_length_sec * fs)

        result["n_fft"] = self.n_fft
        if self.n_fft is None:
            result["n_fft"] = int(2 ** np.ceil(np.log2(result["win_length"])))

        result["window"] = self.window_fn(result["win_length"])
        return result
