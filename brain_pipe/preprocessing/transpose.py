"""Class to transpose data."""
from typing import Union, Dict, Any, Sequence

import numpy as np

from brain_pipe.pipeline.base import PipelineStep


class Transpose(PipelineStep):
    """Transpose data."""

    def __init__(self, keys: Union[Dict[str, Any], Sequence[str], str], **kwargs):
        """Create a transposer.

        Parameters
        ----------
        keys: Union[Dict[str,Any], Sequence[str], str]
            A mapping from the data key to the key of the data to transpose.
            If a string is given, it is used as the key for the data to
            transpose.
        kwargs
        """
        super().__init__(**kwargs)
        self.keys = self.parse_dict_keys(keys)

    def __call__(self, data_dict):
        """Transpose the data.

        Parameters
        ----------
        data_dict: Dict[str, Any]
            The dictionary containing the data to transpose.

        Returns
        -------
        Dict[str, Any]
            The dictionary containing the transposed data.
        """
        for from_key, to_key in self.keys.items():
            data_dict[to_key] = np.transpose(data_dict[from_key])
        return data_dict
