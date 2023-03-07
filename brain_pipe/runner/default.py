"""Default runner for the brain_pipe package."""
import logging
from typing import Callable, Dict, Any, Union, List, Tuple

from brain_pipe.runner.base import Runner
from brain_pipe.dataloaders.base import DataLoader
from brain_pipe.pipeline.base import Pipeline
from brain_pipe.utils.log import default_logging
from brain_pipe.utils.multiprocess import MultiprocessingSingleton


class DefaultRunner(Runner):
    """Default runner for the brain_pipe package."""

    def __init__(
        self,
        nb_processes: int = -1,
        logging_config: Union[Dict[str, Any], Callable, None] = {},
    ):
        """Create the configuration to the runner.

        Parameters
        ----------
        config: Dict[str, Any]
            Configuration dictionary.
        """
        map_fn = MultiprocessingSingleton.get_map_fn(nb_processes)
        super().__init__(map_fn)

        if isinstance(logging_config, dict):
            default_logging(**logging_config)
        elif isinstance(logging_config, Callable):
            logging_config()
        elif logging_config is None:
            logging.root = logging.RootLogger(logging.WARNING)
        else:
            raise ValueError(
                f"Invalid logging config: should be a dict, a callable or None. "
                f"Got {type(logging_config)}."
            )

    def run(self, pipelines_with_loaders: List[Tuple[DataLoader, Pipeline]]):
        """Run the pipelines with their corresponding dataloaders.

        Parameters
        ----------
        pipelines_with_loaders: List[Tuple[DataLoader, Pipeline]]
            A list of tuples containing a DataLoader and its corresponding Pipeline.
        Returns
        -------
        List[Any]
            The results of the pipelines.
        """
        results = super(DefaultRunner, self).run(pipelines_with_loaders)
        MultiprocessingSingleton.clean()
        return results
