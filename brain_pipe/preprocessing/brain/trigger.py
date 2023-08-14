"""Utilities for working with trigger data.

Most of the code in this package is based on our internal EEG processing pipeline
in matlab, EEGORL.
"""

import numpy as np
import scipy.signal
import scipy.interpolate

from brain_pipe.pipeline.base import PipelineStep


def default_drift_correction(
    brain_data,
    brain_trigger_indices,
    brain_fs,
    stimulus_trigger_indices,
    stimulus_fs,
):
    """Correct the drift between the brain response data and the stimulus.

    When the brain response data and the stimulus data are not recorded on
    the same system (i.e. using the same clock), clock drift may cause the
    brain response data to be misaligned with the stimulus. This function
    tries to correct for this by interpolating the brain response data to
    the same length as the stimulus.

    Parameters
    ----------
    brain_data: np.ndarray
        The brain response data. The data is expected to be a 2D array of
        shape (n_channels, n_samples).
    brain_trigger_indices: np.ndarray
        The indices of the brain response data where the triggers are located.
        The data is expected to be a 1D array of integers.
    brain_fs: int
        The sampling frequency of the brain response data.
    stimulus_trigger_indices: np.ndarray
        The indices of the stimulus data where the triggers are located.
        The data is expected to be a 1D array of integers.
    stimulus_fs: int
        The sampling frequency of the stimulus data.

    Returns
    -------
    np.ndarray
        The brain response data with the same length as the stimulus data.
    """
    # We can fix one missing stimulus trigger by adding one to the brain trigger
    if len(brain_trigger_indices) + 1 < len(stimulus_trigger_indices):
        raise ValueError(
            f"Number of triggers does not match "
            f"(in eeg {len(brain_trigger_indices)} were found, "
            f"in stimulus {len(stimulus_trigger_indices)})"
        )
    elif len(brain_trigger_indices) < len(stimulus_trigger_indices):
        # Check if the first trigger is missing
        last_brain_trigger_duration = (
            brain_trigger_indices[-1] - brain_trigger_indices[-2]
        ) / brain_fs
        last_stim_trigger_duration = (
            stimulus_trigger_indices[-1] - stimulus_trigger_indices[-2]
        ) / stimulus_fs
        if (
            (last_brain_trigger_duration * 0.99)
            < last_stim_trigger_duration  # noqa: W503
            < (last_brain_trigger_duration * 1.01)  # noqa: W503
        ):
            # The last trigger is missing
            brain_trigger_indices = np.concatenate(
                (
                    brain_trigger_indices,
                    [
                        brain_trigger_indices[-1]
                        + np.round(  # noqa: W503
                            last_stim_trigger_duration * brain_fs
                        ).astype(int)
                    ],
                ),
                axis=0,
            )
        else:
            # The first trigger is missing
            brain_trigger_indices = np.concatenate(
                (
                    [brain_trigger_indices[0] - brain_fs],
                    brain_trigger_indices,
                ),
                axis=0,
            )
    elif (
        len(brain_trigger_indices) == len(stimulus_trigger_indices) + 1
        and brain_trigger_indices[0] == 0  # noqa: W503
    ):
        # Check if there is an erroneous trigger at the beginning
        brain_trigger_indices = brain_trigger_indices[1:]

    stimulus_diff = stimulus_trigger_indices[-1] - stimulus_trigger_indices[0]
    expected_length = int(np.ceil(stimulus_diff / stimulus_fs * brain_fs))
    real_length = brain_trigger_indices[-1] - brain_trigger_indices[0]
    tmp_eeg = brain_data[:, brain_trigger_indices[0] : brain_trigger_indices[-1]]
    idx_real = np.linspace(0, 1, real_length)
    idx_expected = np.linspace(0, 1, expected_length)
    interpolate_fn = scipy.interpolate.interp1d(idx_real, tmp_eeg, "linear", axis=1)
    new_eeg = interpolate_fn(idx_expected)

    new_start = brain_trigger_indices[0]
    begin_eeg = brain_data[:, :new_start]
    end_eeg = brain_data[:, brain_trigger_indices[-1] + 1 :]

    new_end = int(brain_trigger_indices[-1] + 2 * brain_fs)
    # Make length multiple of samplerate
    new_end = int(np.ceil((new_end - new_start) / brain_fs) * brain_fs + new_start - 1)

    total_eeg = begin_eeg[:, new_start:]
    new_eeg_start = max(new_start - begin_eeg.shape[1], 0)
    new_eeg_end = min(new_end - begin_eeg.shape[1], new_eeg.shape[1])
    total_eeg = np.concatenate(
        (total_eeg, new_eeg[:, new_eeg_start:new_eeg_end]), axis=1
    )
    end_eeg_start = max(new_start - begin_eeg.shape[1] - new_eeg.shape[1], 0)
    end_eeg_end = min(new_end - begin_eeg.shape[1] - new_eeg.shape[1], end_eeg.shape[1])
    total_eeg = np.concatenate(
        (total_eeg, end_eeg[:, end_eeg_start:end_eeg_end]), axis=1
    )
    if total_eeg.shape[1] % brain_fs != 0:
        nb_seconds = np.floor(brain_data.shape[1] / brain_fs)
        total_eeg = total_eeg[:, : int(nb_seconds * brain_fs)]
    return total_eeg


class AlignPeriodicBlockTriggers(PipelineStep):
    """Align the trigger pulses in the form of periodic blocks."""

    def __init__(
        self,
        brain_trigger_processing_fn=lambda x: x,
        postprocessing_fn=default_drift_correction,
        data_key="data",
        data_trigger_key="trigger_data",
        data_sampling_rate_key="eeg_sfreq",
        stimulus_trigger_data_key="trigger_data",
        stimulus_trigger_sampling_rate_key="trigger_sr",
    ):
        """Create a new MatchTriggersToStimulus instance.

        Parameters
        ----------
        data_key: str
        data_trigger_key: str
        data_sampling_rate_key: str
        stimulus_trigger_data_key: str
        stimulus_trigger_sampling_rate_key: str
        """
        super().__init__()
        self.brain_trigger_processing_fn = brain_trigger_processing_fn
        self.postprocessing_fn = postprocessing_fn
        self.data_key = data_key
        self.eeg_trigger_key = data_trigger_key
        self.eeg_sampling_rate_key = data_sampling_rate_key
        self.stimulus_trigger_data_key = stimulus_trigger_data_key
        self.stimulus_trigger_sampling_rate_key = stimulus_trigger_sampling_rate_key

    def split_epochs(self, brain_trigger_indices, brain_fs, nb_epochs):
        """Split the EEG data into epochs.

        Parameters
        ----------
        brain_trigger_indices: np.ndarray
            Indices of the triggers in the EEG data.
        nb_epochs: int
            Number of epochs to split the EEG data into.

        Returns
        -------
        Sequence[np.ndarray]
            EEG data split into epochs.
        """
        trigger_indices = brain_trigger_indices
        # Very unlikely to have a trigger at the exact beginning of the recording
        if trigger_indices[0] == 0:
            trigger_indices = trigger_indices[1:]
        indices_sq = np.argsort(np.diff(trigger_indices))[::-1]
        selected_indices_sq = indices_sq[:nb_epochs]
        sorted_indices = np.sort(selected_indices_sq)
        return np.array_split(brain_trigger_indices, sorted_indices)

    def get_trigger_indices(self, triggers: np.ndarray) -> np.ndarray:
        """Get the indices of the triggers.

        Parameters
        ----------
        triggers: np.ndarray
            Raw trigger data. Should be a 1D array of 0s and 1s.

        Returns
        -------
        np.ndarray
            Indices of the triggers.
        """
        all_indices = np.where(triggers > 0.5)[0]
        diff_trigger_indices = all_indices[1:] - all_indices[:-1]

        # Keep only the gaps between triggers, not the duration of triggers
        indices_to_keep = diff_trigger_indices > 1
        # Assumption that the EEG doesn't start with a trigger
        # in the first sample (shouldn't be the case)
        return all_indices[np.concatenate(([True], indices_to_keep))]

    def __call__(self, data_dict):
        """Match stimulus triggers to triggers from the brain response data.

        Parameters
        ----------
        data_dict: Dict[str, Any]
            Dictionary containing the data to be preprocessed.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the preprocessed data.
        """
        raw_brain_trigger = data_dict[self.eeg_trigger_key]
        brain_fs = data_dict[self.eeg_sampling_rate_key]
        brain_data = data_dict[self.data_key]

        # Process the brain trigger data
        brain_trigger = self.brain_trigger_processing_fn(raw_brain_trigger)
        brain_trigger_indices = self.get_trigger_indices(brain_trigger)

        eeg_epochs = []
        for epoch_index, stimulus_dict in enumerate(data_dict["stimuli"]):
            stimulus_trigger = stimulus_dict[self.stimulus_trigger_data_key]
            stimulus_fs = stimulus_dict[self.stimulus_trigger_sampling_rate_key]
            stimulus_trigger_indices = self.get_trigger_indices(stimulus_trigger)

            new_eeg = self.postprocessing_fn(
                brain_data,
                brain_trigger_indices,
                brain_fs,
                stimulus_trigger_indices,
                stimulus_fs,
            )

            eeg_epochs += [new_eeg]

        data_dict[self.data_key] = eeg_epochs
        return data_dict
