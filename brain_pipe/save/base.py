"""Base class for all save steps."""
import abc
from typing import Any, Dict

from brain_pipe.pipeline.base import PipelineStep


class Save(PipelineStep, abc.ABC):
    """Base class for all save steps."""

    def __init__(self, clear_output=False, overwrite=False, *args, **kwargs):
        """Create a new Save instance.

        Parameters
        ----------
        clear_output: bool
            Whether to clear the output data_dict after saving. This can save space
            when save is the last step in a pipeline.
        overwrite: bool
            Whether to overwrite existing files.
        args: Sequence[Any]
            Additional positional arguments for PipelineStep.
        kwargs: Dict[Any, Any]
            Additional keyword arguments for PipelineStep.
        """
        super().__init__(*args, **kwargs)
        self.clear_output = clear_output
        self.overwrite = overwrite

    @abc.abstractmethod
    def is_already_done(self, data_dict: Dict[str, Any]) -> bool:
        """Check if the step was already done.

        Parameters
        ----------
        data_dict: Dict[str, Any]
            The data dict containing the data to save.

        Returns
        -------
        bool
            True if the step was already done, False otherwise.
        """
        pass

    @abc.abstractmethod
    def is_reloadable(self, data_dict: Dict[str, Any]) -> bool:
        """Check if the data can be reloaded.

        Parameters
        ----------
        data_dict: Dict[str, Any]
            The data dict containing the data to save.

        Returns
        -------
        bool
            True if the data can be reloaded, False otherwise.
        """
        pass

    @abc.abstractmethod
    def reload(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Reload the data.

        Parameters
        ----------
        data_dict: Dict[str, Any]
            The data dict containing the data to save.

        Returns
        -------
        Dict[str, Any]
            The data dict containing the reloaded data.
        """
        pass
