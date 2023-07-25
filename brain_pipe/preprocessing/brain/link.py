"""Link stimulus to brain response data."""
import csv
import os
import pathlib
from typing import Sequence, Dict, Any, Optional, Union, Callable

from brain_pipe.pipeline.base import PipelineStep
from brain_pipe.utils.list import flatten
from brain_pipe.utils.multiprocess import MultiprocessingSingleton
from brain_pipe.utils.path import BIDSStimulusGrouper


class BIDSStimulusInfoExtractor:
    """Extract BIDS compliant stimulus information from an events.tsv file."""

    def __init__(
        self, brain_path_key: str = "data_path", event_info_key: str = "event_info"
    ):
        """Create a new BIDSStimulusInfoExtractor instance.

        Parameters
        ----------
        brain_path_key: str
            The key of the brain data path in the data dict.
        event_info_key: str
            The key store the event information in the data dict.
        """
        self.brain_path_key = brain_path_key
        self.event_info_key = event_info_key

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
        path = brain_dict[self.brain_path_key]
        # Find BIDS compliant events
        events_path = "_".join(path.split("_")[:-1]) + "_events.tsv"
        # Read events
        event_info = self.read_events(events_path)
        brain_dict[self.event_info_key] = event_info
        return event_info

    def read_events(self, events_path: str):
        """Read events from a BIDS compliant events.tsv file.

        Parameters
        ----------
        events_path: str
            The path to the events.tsv file.

        Returns
        -------
        Sequence[Dict[str, Any]]
            The extracted event information. Each dict contains the information
            of one row in the events.tsv file
        """
        with open(events_path) as fp:
            reader = csv.DictReader(fp, dialect="excel-tab")
            event_info = []
            for row in reader:
                event_info += [row]
        return event_info


class BasenameComparisonFn:
    """Compare the basename of the stimulus path to the brain data path."""

    def __init__(
        self, stim_path_key: str = "stimulus_path", ignore_extension: bool = False
    ):
        """Create a new BasenameComparisonFn instance.

        Parameters
        ----------
        stim_path_key: str
            The key of the stimulus path in the data dict. Defaults to
            "stimulus_path".
        ignore_extension: bool
            Whether to ignore the extension of the stimulus path. Defaults to
            False.
        """
        self.stim_path_key = stim_path_key
        self.ignore_extension = ignore_extension

    def process_path(self, path: Any) -> Optional[str]:
        """Process a path so that it can be compared to the brain data path.

        Parameters
        ----------
        path: Any
            The path to process.

        Returns
        -------
        Optional[str]
            The processed path or None if the path is not valid.
        """
        if not isinstance(path, (str, pathlib.Path)):
            return None
        if self.ignore_extension:
            return ".".join(os.path.basename(path).split(".")[:-1])
        return path

    def __call__(
        self, extracted_stim_info: Sequence[Dict[str, Any]], stim_dict: Dict[str, Any]
    ):
        """Compare the basename of the stimulus path to the brain data path.

        Parameters
        ----------
        extracted_stim_info: Sequence[Dict[str, Any]]
            The extracted stimulus information.
        stim_dict
            The dictionary containing the stimulus information.

        Returns
        -------
        bool
            True if the stimulus path matches the brain data path, False
            otherwise.
        """
        all_values = flatten([list(x.values()) for x in extracted_stim_info])
        all_processed_keys = [self.process_path(value) for value in all_values]
        all_processed_keys = [x for x in all_processed_keys if x is not None]
        stimulus_name = self.process_path(stim_dict[self.stim_path_key])
        return stimulus_name in all_processed_keys


def default_multiprocessing_key_fn(data_dict):
    """Create a multiprocessing key from a data dict.

    This is used to check during multiprocessing if the stimulus is currently
    being processed by another process.

    Parameters
    ----------
    data_dict: Dict[str, Any]
        The data dict to create the key from.

    Returns
    -------
    str
        The multiprocessing key.
    """
    return str(data_dict["stimulus_path"])


class LinkStimulusToBrainResponse(PipelineStep):
    """Link stimulus to Brain data."""

    multiprocessing_dict = MultiprocessingSingleton.manager.dict()
    multiprocessing_condition = MultiprocessingSingleton.manager.Condition()

    def __init__(
        self,
        stimulus_data: Union[Sequence[Dict[str, Any]], PipelineStep],
        extract_stimuli_information_fn=BIDSStimulusInfoExtractor(),
        comparison_fn=BasenameComparisonFn(),
        stimulus_path_key="stimulus_path",
        stimuli_key="stimuli",
        grouper: Optional[BIDSStimulusGrouper] = None,
        key_fn_for_multiprocessing=default_multiprocessing_key_fn,
        *args,
        **kwargs,
    ):
        """Create a new LinkStimulusToEEG instance.

        Parameters
        ----------
        stimulus_data_dicts: Sequence[Dict[str, Any]]
            A sequence of data dicts containing the stimulus data.
        find_stimulus_fn: Callable[[str], Sequence[str]]
            A function that takes the path to the EEG recording and returns
            a sequence of corresponding stimulus paths.
        comparison_fn: Callable[[Dict[str, Any], str], bool]
            A function that takes a data dict and a stimulus path and returns
            True if the data dict corresponds to the stimulus path.
        stimulus_path_key: str
            The key in the data dict that contains the path to the stimulus.
        stimuli_key: str
            The key in the data dict that contains the stimulus data.
        pipeline_cache: Optional[PipelineCache]
            The pipeline cache to use to load the stimulus data dict if
            necessary. This is only necessary if the stimulus data dicts
            are cached.
        """
        super(LinkStimulusToBrainResponse, self).__init__(*args, **kwargs)
        self.stimulus_data = stimulus_data
        self.extract_stimuli_information_fn = extract_stimuli_information_fn
        self.comparison_fn = comparison_fn
        self.stimulus_path_key = stimulus_path_key
        self.stimuli_key = stimuli_key
        if grouper is None and isinstance(stimulus_data, Callable):
            raise ValueError(
                f"`grouper` must be set for {self.__class__.__name__} if stimulus "
                "data is a preprocessing step/callable."
            )
        self.grouper = grouper
        self.key_fn_for_multiprocessing = key_fn_for_multiprocessing

    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Link the stimulus to the EEG data.

        Parameters
        ----------
        data_dict: Dict[str, Any]
            The data dict containing the EEG data.

        Returns
        -------
        Dict[str, Any]
            The data dict with the corresponding stimulus data added.
        """
        data_dict = super(LinkStimulusToBrainResponse, self).__call__(data_dict)
        # Find the corresponding stimuli data_dicts
        stimulus_info_from_brain = self.extract_stimuli_information_fn(data_dict)
        all_stimuli = []
        if isinstance(self.stimulus_data, Sequence):
            for stimulus_dict in self.stimulus_data:
                if self.comparison_fn(stimulus_info_from_brain, stimulus_dict):
                    all_stimuli += [stimulus_dict]
        else:
            for stim_info in stimulus_info_from_brain:
                prototype_stim_dict = self.grouper(stim_info)
                key = self.key_fn_for_multiprocessing(prototype_stim_dict)
                with self.multiprocessing_condition:
                    # Check if no other processes are already running this
                    while key in self.multiprocessing_dict:
                        # Wait for the process to finish
                        self.multiprocessing_condition.wait()
                    self.multiprocessing_dict[key] = True
                try:
                    stimulus_dicts = self.stimulus_data(prototype_stim_dict)
                finally:
                    # Notify all waiting processes of that this is done
                    with self.multiprocessing_condition:
                        # Remove the key from the multiprocessing dict to signal that
                        # this specific stimulus is processed
                        del self.multiprocessing_dict[key]
                        self.multiprocessing_condition.notify_all()
                if isinstance(stimulus_dicts, dict):
                    stimulus_dicts = [stimulus_dicts]
                all_stimuli += stimulus_dicts

        data_dict[self.stimuli_key] = all_stimuli
        return data_dict
