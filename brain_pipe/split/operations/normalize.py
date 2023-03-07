"""Normalize data."""
import numpy as np

from brain_pipe.split.operations.base import SplitterOperation


class Standardize(SplitterOperation):
    r"""Standardize data.

    .. math::

        x_standardized = \frac{x - \mu}{\sigma}

    """

    def __init__(self, axis=0):
        """Create a standardizer.

        Parameters
        ----------
        axis: int
            Axis to standardize the data on.
        """
        self.axis = axis
        self.mean = None
        self.std = None

    def __call__(self, data):
        """Standardize the data.

        Parameters
        ----------
        data: np.ndarray
            Data to standardize.

        Returns
        -------
        np.ndarray
            Standardized data.
        """
        if self.mean is None:
            self.mean = np.mean(data, axis=self.axis, keepdims=True)
            self.std = np.std(data, axis=self.axis, keepdims=True)
        return (data - self.mean) / self.std

    def reset(self):
        """Reset the standardizer."""
        self.mean = None
        self.std = None
