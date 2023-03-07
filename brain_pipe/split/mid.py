"""Splitting data in sets in the middle of the data."""
import numpy as np

from brain_pipe.split.base import Splitter


class MidSplit(Splitter):
    """Split data in the middle of the data."""

    def __init__(self, *args, **kwargs):
        """Create a mid-splitter.

        Parameters
        ----------
        args: Sequence[Any]
            Arguments to pass to the Splitter.
        kwargs: Dict[str, Any]
            Keyword arguments to pass to the Splitter.
        """
        super().__init__(*args, **kwargs)
        self._sort_split_names_fractions()

    def _sort_split_names_fractions(self):
        split_name_fraction = zip(self.split_names, self.split_fractions)
        new_names = []
        new_fractions = []
        for name, fraction in sorted(split_name_fraction, key=lambda x: x[1]):
            new_names += [name]
            new_fractions += [fraction]
        self.split_names = new_names
        self.split_fractions = new_fractions

    def split(self, data, shortest_length, split_fraction, start_index):
        """Split the data into sets.

        The largest set will be extracted from the front and back of the data
        (this should also be the first set that this method encounters,
        see :meth:`sort_split_names_fractions`)
        The other sets will be extracted from the middle of the data, sequentially.

        Parameters
        ----------
        data: np.ndarray
            Data to split.
        shortest_length: int
            Length of the shortest data.
        split_fraction: float
            Fraction of the data to split into the current set.
        start_index: int
            Index to start splitting the data from.

        Returns
        -------
        Tuple[np.ndarray, int]
            The split data and the index to start splitting the next data from.
        """
        max_fraction = np.max(self.split_fractions)
        data_length = data.shape[self.axis]
        # Biggest fraction should be split from the front and back of the data
        if split_fraction == max_fraction:
            end_index_1 = int(np.round(shortest_length * split_fraction / 2))
            end_index_2 = data_length - end_index_1
            split_data = np.take(
                data,
                np.concatenate(
                    (
                        np.arange(0, end_index_1),
                        np.arange(data_length - end_index_2, data_length),
                    ),
                    axis=self.axis,
                ),
                axis=self.axis,
            )
            return split_data, end_index_1
        # Smaller fractions should be split from the middle of the data
        else:
            end_index = start_index + int(np.round(shortest_length * split_fraction))
            split_data = np.take(
                data, np.arange(start_index, end_index), axis=self.axis
            )
            return split_data, end_index
