"""Run the pipeline on the auditory EEG dataset from the auditory EEG dataset example."""
import argparse
import datetime
import glob
import gzip
import logging
import os
import pathlib
import pickle
import time
from typing import Any, Dict, Union, Sequence

import librosa
import numpy as np
import scipy.signal

from brain_pipe.dataloaders.path import GlobLoader
from brain_pipe.pipeline.default import DefaultPipeline
from brain_pipe.preprocessing.brain.artifact import (
    InterpolateArtifacts,
    ArtifactRemovalMWF,
)
from brain_pipe.preprocessing.brain.eeg.biosemi import (
    biosemi_trigger_processing_fn,
)
from brain_pipe.preprocessing.brain.eeg.load import LoadEEGNumpy
from brain_pipe.preprocessing.brain.epochs import SplitEpochs
from brain_pipe.preprocessing.brain.link import (
    LinkStimulusToBrainResponse,
    BIDSStimulusInfoExtractor,
)
from brain_pipe.preprocessing.brain.rereference import CommonAverageRereference
from brain_pipe.preprocessing.brain.trigger import (
    AlignPeriodicBlockTriggers,
)
from brain_pipe.preprocessing.filter import SosFiltFilt
from brain_pipe.preprocessing.resample import ResamplePoly
from brain_pipe.save.default import DefaultSave
from brain_pipe.preprocessing.stimulus.audio.envelope import GammatoneEnvelope
from brain_pipe.preprocessing.stimulus.load import LoadStimuli
from brain_pipe.utils.log import default_logging, DefaultFormatter
from brain_pipe.utils.multiprocess import MultiprocessingSingleton
from brain_pipe.utils.path import BIDSStimulusGrouper


def default_is_trigger_fn(path: Union[str, pathlib.Path]):
    return os.path.basename(path).startswith("t_")


def default_is_noise_fn(path: Union[str, pathlib.Path]):
    return os.path.basename(path).startswith("noise_")


def default_is_video_fn(path: Union[str, pathlib.Path]):
    return os.path.basename(path).startswith("VIDEO")


def default_key_fn(path: Union[str, pathlib.Path]):
    basename = os.path.basename(path)
    for prefix in ("t_", "noise_"):
        if basename.startswith(prefix):
            return prefix.join(basename.split(prefix)[1:])
    return basename


class StimulusGrouper:
    def __init__(
        self,
        key_fn=default_key_fn,
        is_trigger_fn=default_is_trigger_fn,
        is_noise_fn=default_is_noise_fn,
        is_video_fn=default_is_video_fn,
        filter_no_triggers=True,
    ):
        self.key_fn = key_fn
        self.is_trigger_fn = is_trigger_fn
        self.is_noise_fn = is_noise_fn
        self.is_video_fn = is_video_fn
        self.filter_no_triggers = filter_no_triggers

    def _postprocess(self, data_dicts):
        new_data_dicts = []
        for data_dict in data_dicts.values():
            if data_dict["stimulus_path"] is None:
                if data_dict["trigger_path"] is None:
                    raise ValueError(
                        "Found data dict without stimulus and trigger, "
                        f"which should not be possible: {data_dict}"
                    )
                else:
                    logging.warning(
                        f"Found a data_dict with no stimulus: {data_dict}. "
                        f"This is fine if the data was collected in silence. "
                        f"Otherwise, adapt the `key_fn` and/or the `is_*_fn` "
                        f"of the StimulusGrouper."
                    )

            if data_dict["trigger_path"] is None:
                logging.error(
                    f"No trigger path found for {data_dict['stimulus_path']}."
                    f"If a trigger path shoud be present, adapt the `key_fn` "
                    f"and/or the `is_*_fn` of the StimulusGrouper."
                )
                if self.filter_no_triggers:
                    logging.error(
                        f"\tFiltering out stimulus data for "
                        f"{data_dict['stimulus_path']}"
                    )
                    continue
            new_data_dicts += [data_dict]
        return new_data_dicts

    def __call__(self, files: Sequence[Union[str, pathlib.Path]]) -> Sequence[Dict]:
        data_dicts = {}
        for path in files:
            keys = self.key_fn(path)
            if isinstance(keys, str):
                keys = [keys]
            for key in keys:
                if key not in data_dicts:
                    data_dicts[key] = {
                        "trigger_path": None,
                        "noise_path": None,
                        "video_path": None,
                        "stimulus_path": None,
                    }
                if self.is_trigger_fn(path):
                    data_dicts[key]["trigger_path"] = path
                elif self.is_noise_fn(path):
                    data_dicts[key]["noise_path"] = path
                elif self.is_video_fn(path):
                    data_dicts[key]["video_path"] = path
                else:
                    data_dicts[key]["stimulus_path"] = path
        logging.info(f"Found {len(data_dicts)} stimulus groups")
        data_dict_list = self._postprocess(data_dicts)
        return data_dict_list


class BIDSAPRStimulusInfoExtractor(BIDSStimulusInfoExtractor):
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
    separator: str
        The separator to use between the different parts of the filename.

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


def sparrkulee_logging(log_path):
    parsed_path = log_path.format(
        datetime=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    handler = logging.FileHandler(parsed_path)

    handler.setLevel(logging.DEBUG)
    handler.setFormatter(DefaultFormatter())
    default_logging(handlers=[handler])


def run_exporl_auditory_eeg_pipeline(
    base_dir,
    root_dir,
    nb_processes=-1,
    overwrite=False,
    log_path="auditory_eeg_dataset.log",
):
    # SUBJECT/STIMULUS SELECTION CRITERIA #
    #######################################
    output_dir = os.path.join(base_dir, "output")
    stimuli_dir = os.path.join(base_dir, "stimuli")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(stimuli_dir, exist_ok=True)

    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(DefaultFormatter())
    default_logging(handlers=[handler])
    map_fn = MultiprocessingSingleton.get_map_fn(nb_processes)

    logging.info("Retrieving BIDS layout...")
    start_time = time.time()
    data_loader = GlobLoader(
        [os.path.join(root_dir, "sub-*", "*", "eeg", "*.bdf*")],
        filter_fns=[lambda x: "restingState" not in x],
        key="data_path",
    )

    # STEPS #
    #########

    stimulus_steps = DefaultPipeline(
        steps=[
            LoadStimuli(load_fn=temp_stimulus_load_fn),
            GammatoneEnvelope(),
            ResamplePoly(64, "envelope_data", "stimulus_sr"),
            DefaultSave(stimuli_dir, overwrite=overwrite),
        ],
        on_error=DefaultPipeline.RAISE,
    )

    eeg_steps = [
        LinkStimulusToBrainResponse(
            stimulus_data=stimulus_steps,
            extract_stimuli_information_fn=BIDSAPRStimulusInfoExtractor(),
            grouper=BIDSStimulusGrouper(
                bids_root=root_dir,
                mapping={"stim_file": "stimulus_path", "trigger_file": "trigger_path"},
                subfolders=["stimuli", "eeg"],
            ),
        ),
        LoadEEGNumpy(unit_multiplier=1e6, channels_to_select=list(range(64))),
        SosFiltFilt(
            scipy.signal.butter(1, 0.5, "highpass", fs=1024, output="sos"),
            emulate_matlab=True,
            axis=1,
        ),
        InterpolateArtifacts(),
        AlignPeriodicBlockTriggers(biosemi_trigger_processing_fn),
        SplitEpochs(),
        ArtifactRemovalMWF(),
        CommonAverageRereference(),
        ResamplePoly(64, axis=1),
        DefaultSave(
            output_dir,
            {"eeg": "data"},
            overwrite=overwrite,
            clear_output=True,
            filename_fn=bids_filename_fn,
        ),
    ]

    ##########################
    # Preprocessing pipeline #
    ##########################

    logging.info("Starting with the EEG preprocessing")
    logging.info("===================================")

    # Create data_dicts for the EEG files
    # Create the EEG pipeline
    eeg_pipeline = DefaultPipeline(steps=eeg_steps)
    # Execute the EEG pipeline
    list(map_fn(eeg_pipeline, data_loader))
    MultiprocessingSingleton.clean()

    # Save the envelopes also in .npy format
    for stim_file in glob.glob(os.path.join(stimuli_dir, "*.data_dict")):
        new_filepath = stim_file.replace(".data_dict", "_envelope.npy")
        with open(stim_file, "rb") as f:
            data_dict = pickle.load(f)
        logging.info(f"Saving envelope data from {stim_file} to {new_filepath}")
        np.save(new_filepath, data_dict["envelope_data"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess the auditory EEG dataset")

    parser.add_argument(
        "--nb_processes",
        type=int,
        default=-1,
        help="Number of processes to use for the preprocessing. "
        "The default is to use all available cores (-1).",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files"
    )
    parser.add_argument(
        "--log_path", type=str, default="auditory_eeg_dataset_{datetime}.log"
    )
    parser.add_argument(
        "download_path",
        type=str,
        help="Path to the folder where the dataset is downloaded",
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Path to the folder where the preprocessed data will be saved",
    )
    args = parser.parse_args(
        [
            "/esat/audioslave/mjalilpo/published_dataset/FINAL_MAPPING/train",
            "/esat/audioslave/baccou/sparkulee_processed_from_scratch/",
        ]
    )
    run_exporl_auditory_eeg_pipeline(
        args.output_path,
        args.download_path,
        args.nb_processes,
        args.overwrite,
        args.log_path.format(
            datetime=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ),
    )
