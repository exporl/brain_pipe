"""Splitters that takes sequential slices of the data."""
import numpy as np

from brain_pipe.split.base import Splitter


class SequentialSplit(Splitter):
    """Splitter that takes sequential slices of the data."""

    def split(self, data, shortest_length, split_fraction, start_index):
        """Split the data into sequential sets.

        Parameters
        ----------
        data: np.ndarray
            Data to split.
        shortest_length: int
            Length of the shortest data.
        split_fraction:
            Fraction of the data to split into the current set.
        start_index: int
            Index to start splitting the data from.

        Returns
        -------
        Tuple[np.ndarray, int]
            The split data and the index to start splitting the next data from.
        """
        end_index = int(np.round(shortest_length * split_fraction))
        split_data = np.take(data, np.arange(start_index, end_index), axis=self.axis)
        return split_data, end_index
