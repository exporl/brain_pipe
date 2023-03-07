"""Utilities for working with BIDS paths."""
import glob
import logging
import os
from typing import Sequence


class BIDSPathGenerator:
    """Generate BIDS paths for a given root directory."""

    def __init__(self, root_dir):
        """Create a new BIDSPathGenerator.

        Parameters
        ----------
        root_dir: str
            The root directory of the BIDS dataset.
        """
        self.root_dir = root_dir

    def _parse_part(self, part):
        # Select everything when part is None.
        if part is None:
            return ["*"]
        elif isinstance(part, str):
            return [part]
        elif isinstance(part, Sequence):
            return part
        else:
            raise ValueError(f"Invalid part for BIDS path: {part}")

    def select_paths(
        self,
        subjects=None,
        sessions=None,
        tasks=None,
        runs=None,
        extensions="eeg",
        suffix="bdf",
    ):
        """Select BIDS paths for a given set of parameters.

        Parameters
        ----------
        subjects: Optional[Union[str, Sequence[str]]]
            The subjects to select. When None, all subjects are selected.
            When a string, only the subject with the given name is selected.
            When a sequence of strings, all subjects with the given names are
            selected.
        sessions: Optional[Union[str, Sequence[str]]]
            The sessions to select. When None, all sessions are selected.
            When a string, only the session with the given name is selected.
            When a sequence of strings, all sessions with the given names are
            selected.
        tasks: Optional[Union[str, Sequence[str]]]
            The tasks to select. When None, all tasks are selected.
            When a string, only the task with the given name is selected.
            When a sequence of strings, all tasks with the given names are
            selected.
        runs: Optional[Union[str, Sequence[str]]]
            The runs to select. When None, all runs are selected.
            When a string, only the run with the given name is selected.
            When a sequence of strings, all runs with the given names are
            selected.
        extensions: Optional[Union[str, Sequence[str]]]
            The extensions to select. When None, all extensions are selected.
            When a string, only the extension with the given name is selected.
            When a sequence of strings, all extensions with the given names are
            selected.
        suffix: Optional[Union[str, Sequence[str]]]
            The suffixes to select. When None, all suffixes are selected.
            When a string, only the suffix with the given name is selected.
            When a sequence of strings, all suffixes with the given names are
            selected.

        Returns
        -------
        List[str]
            A list of paths that match the given parameters.
        """
        paths = []

        for subject in self._parse_part(subjects):
            for session in self._parse_part(sessions):
                for task in self._parse_part(tasks):
                    for run in self._parse_part(runs):
                        for extension in self._parse_part(extensions):
                            for suffix in self._parse_part(suffix):
                                search_path = os.path.join(
                                    self.root_dir,
                                    f"sub-{subject}",
                                    f"ses-{session}",
                                    f"{extension}",
                                    f"sub-{subject}_ses-{session}_task-{task}_"
                                    f"run-{run}_{extension}.{suffix}",
                                )
                                paths += glob.glob(search_path)
                                # session is not required in BIDS
                                if session == "*":
                                    search_path = os.path.join(
                                        self.root_dir,
                                        f"sub-{subject}",
                                        f"{extension}",
                                        f"sub-{subject}_task-{task}_run-{run}_"
                                        f"{extension}.{suffix}",
                                    )
                                    paths += glob.glob(search_path)
        return paths


class BIDSStimulusGrouper:
    """Group stimulus files by subject, session, task and run."""

    def __init__(
        self,
        bids_root,
        mapping={"stim_file": "stimulus_path"},
        subfolders=["stimuli"],
        na_values=["n/a"],
    ):
        """Create a new BIDSStimulusGrouper.

        Parameters
        ----------
        bids_root: str
            The root directory of the BIDS dataset.
        mapping: Dict[str, str]
            A mapping from the column names in the events.tsv file to the
            column names in the stimulus dictionary.
        subfolders: Sequence[str]
            The subfolders in which the stimulus files are located.
        na_values: Sequence[str]
            The values that should be interpreted as missing values.
        """
        self.bids_root = bids_root
        self.mapping = mapping
        self.subfolders = subfolders
        self.na_values = na_values

    def __call__(self, events_row):
        """Group stimulus files by subject, session, task and run.

        Parameters
        ----------
        events_row: Dict[str, str]
            A row from the events.tsv file.

        Returns
        -------
        Dict[str, str]
            A dictionary with the stimulus files grouped by subject, session,
        """
        stimulus_dict = {}
        for from_key, to_key in self.mapping.items():
            if from_key not in events_row:
                logging.warning(
                    f"Could not find {from_key} in events row, skipping stimulus "
                    f"file {events_row}"
                )
                stimulus_dict[to_key] = None
                continue
            events_item = events_row[from_key]
            if events_item in self.na_values:
                stimulus_dict[to_key] = None
                continue

            subfolders = [
                folder
                for folder in self.subfolders
                if folder not in events_item.split("/")
            ]

            stimulus_dict[to_key] = os.path.join(
                self.bids_root, *subfolders, events_item
            )
        return stimulus_dict
