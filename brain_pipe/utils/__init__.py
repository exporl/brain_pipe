"""General utilities."""
import abc
from typing import Dict


class ObjectsWithConfig(abc.ABC):
    """Base class for objects that can be configured with a dictionary."""

    @abc.abstractmethod
    def apply_config(self, config: Dict):
        """Apply the configuration to the object.

        Parameters
        ----------
        config: Dict
            The configuration to apply.
        """
        pass
