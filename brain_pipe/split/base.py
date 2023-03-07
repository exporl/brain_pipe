"""Code to split data into sets."""
import abc
from typing import Optional, Sequence, Union, Dict, Any, Tuple

import numpy as np

from brain_pipe.pipeline.base import PipelineStep
from brain_pipe.split.operations.base import SplitterOperation


class Splitter(PipelineStep, abc.ABC):
    """Base class for splitting data into sets."""

    def __init__(
        self,
        feature_mapping: Union[Dict[str, Any], Sequence[str], str],
        split_fractions: Sequence[Union[int, float]],
        split_names: Sequence[str],
        extra_operation: Optional[SplitterOperation] = None,
        axis=0,
    ):
        """Create a splitter.

        Parameters
        ----------
        feature_mapping: Union[Dict[str, Any], Sequence[str], str]
            A mapping from the data key to the key of the data to split.
        split_fractions: Sequence[Union[int, float]]
            Fractions of the data to split into the different sets.
        split_names: Sequence[str]
            Names of the different sets.
        extra_operation: Optional[SplitterOperation]
            Operation to perform on the split data. If None, no operation is
            performed.
        axis: int
            Axis to split the data on.
        """
        self.feature_mapping = self.parse_dict_keys(feature_mapping)
        self.split_fractions = self._normalize_split_fraction(split_fractions)
        self.split_names = split_names
        self.extra_operation = extra_operation
        self.axis = axis

    def _normalize_split_fraction(self, split_fractions):
        return [fraction / sum(split_fractions) for fraction in split_fractions]

    @abc.abstractmethod
    def split(
        self, data: Any, shortest_length: int, split_fraction: float, start_index: int
    ) -> Tuple[Any, int]:
        """Split the data into sets.

        Parameters
        ----------
        data: Any
            Data to split.
        shortest_length: int
            Length of the shortest data.
        split_fraction: float
            Fraction of the data to split into the current set.
        start_index: int
            Index to start splitting the data from.

        Returns
        -------
        Any, int
            The split data and the index to start splitting the next data from.
        """
        pass

    def __call__(self, data_dict):
        """Split data into sets.

        Parameters
        ----------
        data_dict: Dict[str, Any]
            The data dict containing the data to split.

        Returns
        -------
        Dict[str, Any]
            The data dict containing the split data.
        """
        shortest_length = min(
            [data_dict[key].shape[self.axis] for key in self.feature_mapping.keys()]
        )
        for from_key, to_key in self.feature_mapping.items():
            data = np.take(
                data_dict[from_key], np.arange(0, shortest_length), axis=self.axis
            )
            resulting_data = {}

            self.extra_operation.reset()
            start_index = 0
            for split_name, split_fraction in zip(
                self.split_names, self.split_fractions
            ):
                split_data, start_index = self.split(
                    data, shortest_length, split_fraction, start_index
                )
                if self.extra_operation is not None:
                    split_data = self.extra_operation(split_data)
                resulting_data[split_name] = split_data
            del data_dict[from_key]
            data_dict[to_key] = resulting_data

        return data_dict
