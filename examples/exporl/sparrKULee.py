"""Run the default preprocessing pipeline on sparrKULee."""
import argparse
import datetime
import logging
import os

import scipy.signal
import scipy.signal.windows

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
)
from brain_pipe.preprocessing.brain.rereference import CommonAverageRereference
from brain_pipe.preprocessing.brain.trigger import (
    AlignPeriodicBlockTriggers,
)
from brain_pipe.preprocessing.filter import SosFiltFilt
from brain_pipe.preprocessing.resample import ResamplePoly
from brain_pipe.preprocessing.stimulus.audio.envelope import GammatoneEnvelope
from brain_pipe.preprocessing.stimulus.audio.spectrogram import LibrosaMelSpectrogram
from brain_pipe.preprocessing.stimulus.load import LoadStimuli
from brain_pipe.runner.default import DefaultRunner
from brain_pipe.save.default import DefaultSave
from brain_pipe.utils.log import default_logging, DefaultFormatter
from brain_pipe.utils.path import BIDSStimulusGrouper
from examples.exporl.utils import temp_stimulus_load_fn, SparrKULeeSpectrogramKwargs, \
    BIDSAPRStimulusInfoExtractor, bids_filename_fn


def run_preprocessing_pipeline(
    root_dir,
    preprocessed_stimuli_dir,
    preprocessed_eeg_dir,
    nb_processes=-1,
    overwrite=False,
    log_path="sparrKULee.log",
):
    """Construct and run the preprocessing on SparrKULee.

    Parameters
    ----------
    root_dir: str
        The root directory of the dataset.
    preprocessed_stimuli_dir:
        The directory where the preprocessed stimuli should be saved.
    preprocessed_eeg_dir:
        The directory where the preprocessed EEG should be saved.
    nb_processes: int
        The number of processes to use. If -1, the number of processes is
        automatically determined.
    overwrite: bool
        Whether to overwrite existing files.
    log_path: str
        The path to the log file.
    """
    #########
    # PATHS #
    #########
    os.makedirs(preprocessed_eeg_dir, exist_ok=True)
    os.makedirs(preprocessed_stimuli_dir, exist_ok=True)

    ###########
    # LOGGING #
    ###########
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(DefaultFormatter())
    default_logging(handlers=[handler])

    ################
    # DATA LOADING #
    ################
    logging.info("Retrieving BIDS layout...")
    data_loader = GlobLoader(
        [os.path.join(root_dir, "sub-*", "*", "eeg", "*.bdf*")],
        filter_fns=[lambda x: "restingState" not in x],
        key="data_path",
    )

    #########
    # STEPS #
    #########

    stimulus_steps = DefaultPipeline(
        steps=[
            LoadStimuli(load_fn=temp_stimulus_load_fn),
            GammatoneEnvelope(),
            LibrosaMelSpectrogram(
                power_factor=0.6, librosa_kwargs=SparrKULeeSpectrogramKwargs()
            ),
            ResamplePoly(64, "envelope_data", "stimulus_sr"),
            # Comment out the next line if you don't want to use mel
            DefaultSave(
                preprocessed_stimuli_dir,
                to_save={
                    "envelope": "envelope_data",
                    # Comment out the next line if you don't want to use mel
                    "mel": "spectrogram_data",
                },
                overwrite=overwrite,
            ),
            DefaultSave(preprocessed_stimuli_dir, overwrite=overwrite),
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
            preprocessed_eeg_dir,
            {"eeg": "data"},
            overwrite=overwrite,
            clear_output=True,
            filename_fn=bids_filename_fn,
        ),
    ]

    #########################
    # RUNNING THE PIPELINE  #
    #########################

    logging.info("Starting with the EEG preprocessing")
    logging.info("===================================")

    # Create data_dicts for the EEG files
    # Create the EEG pipeline
    eeg_pipeline = DefaultPipeline(steps=eeg_steps)

    DefaultRunner(
        nb_processes=nb_processes,
        logging_config=lambda: None,
    ).run(
        [(data_loader, eeg_pipeline)],
    )


if __name__ == "__main__":
    # Code for the sparrKULee dataset
    # (https://rdr.kuleuven.be/dataset.xhtml?persistentId=doi:10.48804/K3VSND)
    #
    # A slight adaption of this code can also be found in the spaRRKULee repository:
    # https://github.com/exporl/auditory-eeg-dataset
    # under preprocessing_code/sparrKULee.py

    # Set the default log folder
    default_log_folder = os.path.dirname(os.path.abspath(__file__))

    # Parse arguments from the command line
    parser = argparse.ArgumentParser(description="Preprocess the SparrKULee dataset")
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
        "--log_path",
        type=str,
        default=os.path.join(default_log_folder, "sparrKULee_{datetime}.log"),
    )
    parser.add_argument(
        "--dataset_folder",
        type=str,
        help="Path to the folder where the dataset is downloaded",
    )
    parser.add_argument(
        "--preprocessed_stimuli_path",
        type=str,
        help="Path to the folder where the preprocessed stimuli will be saved",
    )
    parser.add_argument(
        "--preprocessed_eeg_path",
        type=str,
        help="Path to the folder where the preprocessed EEG will be saved",
    )
    args = parser.parse_args()

    # Run the preprocessing pipeline
    run_preprocessing_pipeline(
        args.dataset_folder,
        args.preprocessed_stimuli_path,
        args.preprocessed_eeg_path,
        args.nb_processes,
        args.overwrite,
        args.log_path.format(
            datetime=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ),
    )
