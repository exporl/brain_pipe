"""Code to calculate Spectrograms."""
import typing

import librosa
import numpy as np

from brain_pipe.pipeline.base import PipelineStep


class LibrosaMelSpectrogram(PipelineStep):
    """Calculates mel spectrogram using librosa.

    Code was based on the ICASSP 2023 auditory EEG challenge:
    https://github.com/exporl/auditory-eeg-challenge-2023-code/blob/main/
    task1_match_mismatch/util/mel_spectrogram.py
    """

    def __init__(
        self,
        stimulus_data_key="stimulus_data",
        stimulus_sr_key="stimulus_sr",
        output_key="spectrogram_data",
        output_sr_key="spectrogram_sr",
        power_factor=1.0,
        sort_fn=None,
        librosa_kwargs={},
        **kwargs,
    ):
        """Calculate the mel spectrogram of a raw speech file.

        Parameters
        ---------
        stimulus_data_key : str
            The key in the data dictionary that contains the stimulus data
        stimulus_sr_key : str
            The key in the data dictionary that contains the stimulus sampling rate
        output_key : str
            The key in the data dictionary to store the spectrogram
        power_factor: float
            The power factor for each sample
        sort_fn : Callable
            A function to sort the kwargs for librosa.feature.melspectrogram when
            when parsing the kwargs. This is useful when the callables are used that
            depend on other kwargs.
        librosa_kwargs : Union[Dict[str, Any], Callable]
            Keyword arguments to pass to librosa.feature.melspectrogram. Can also be
            a callable that takes in data_dict and returns a dict of kwargs.
        kwargs : dict
            Additional keyword arguments for the PipelineStep
        """
        super(LibrosaMelSpectrogram, self).__init__(**kwargs)
        self.stimulus_data_key = stimulus_data_key
        self.stimulus_sr_key = stimulus_sr_key
        self.output_key = output_key
        self.output_sr_key = output_sr_key
        self.sort_fn = sort_fn
        self.power_factor = power_factor
        self.librosa_kwargs = librosa_kwargs

    def parse_librosa_kwargs(self, data_dict):
        """Parse kwargs for Librosa's melspectrogram function.

        Parameters
        ----------
        data_dict: Dict[str, Any]
            The data dictionary

        Returns
        -------
        Dict[str, Any]
            The parsed kwargs for Librosa's melspectrogram function
        """
        # If it is a callable, call it
        if isinstance(self.librosa_kwargs, typing.Callable):
            return self.librosa_kwargs(data_dict)

        # Warning: ordering matters here depending on the what the callable values
        # do
        for key, value in sorted(list(self.librosa_kwargs.items()), key=self.sort_fn):
            if isinstance(value, typing.Callable):
                self.librosa_kwargs[key] = value(self.librosa_kwargs, data_dict)
            else:
                self.librosa_kwargs[key] = value
        return self.librosa_kwargs

    def __call__(self, data_dict):
        """Calculate the mel spectrogram.

        Parameters
        ----------
        data_dict: Dict[str, Any]
            The data dictionary

        Returns
        -------
        Dict[str, Any]
            The data dictionary with the mel spectrogram.
        """
        audio = data_dict[self.stimulus_data_key]
        fs = data_dict[self.stimulus_sr_key]

        librosa_kwargs = self.parse_librosa_kwargs(data_dict)

        # DC removal
        audio = audio - np.mean(audio)
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio, sr=fs, **librosa_kwargs
        ).T
        # Apply power law scaling
        mel_spectrogram = np.power(mel_spectrogram, self.power_factor)

        data_dict[self.output_key] = mel_spectrogram
        # Estimation of the sampling rate of the spectrogram
        data_dict[self.output_sr_key] = fs / librosa_kwargs.get("hop_length", 512)
        return data_dict
