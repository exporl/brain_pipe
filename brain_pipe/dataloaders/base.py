"""Base class to load data from a source."""
import abc
from typing import Iterator


class DataLoader(Iterator, abc.ABC):
    """Base class to load data from a source.

    DataLoaders are used to load data from a source (e.g. a file) and iterate over it.
    Therefore, they have an __iter__ method that should be implemented in subclasses.
    """

    def __init__(self, has_length=True):
        """Initialize the DataLoader."""
        self.it = None
        self._has_length = False
        self.has_length = has_length

    @abc.abstractmethod
    def __iter__(self) -> Iterator:
        """Iterate over the files.

        Returns
        -------
        Iterator[Any]
            An iterator over a dataset.
        """
        pass

    def __next__(self):
        """Load the next file."""
        if self.it is None:
            self.it = self.__iter__()
        return next(self.it)

    def compute_length(self):
        """Compute the length of the DataLoader.

        Returns
        -------
        int
            The length of the DataLoader.

        Notes
        -----
        This function will be added as the __len__ method if :attr:`has_length` is True.
        You can override this func
        """
        return sum(1 for _ in self.__iter__())

    def __len__(self):
        """Compute the length of the DataLoader.

        Returns
        -------
        int
            The length of the DataLoader.

        Raises
        ------
        TypeError
            If :attr:`has_length` is False, computing the length is disabled.
        """
        if self.has_length:
            return self.compute_length()
        else:
            raise TypeError(
                "This DataLoader does not have a length (set has_length to True to "
                "enable computing the length)."
            )
