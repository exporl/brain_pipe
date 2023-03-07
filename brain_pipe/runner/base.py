"""Base Runner class, used to run Pipelines with DataLoaders."""
import logging
import abc
from typing import List, Tuple

from brain_pipe.dataloaders.base import DataLoader
from brain_pipe.pipeline.base import Pipeline


class Runner(abc.ABC):
    """Base Runner class, used to run Pipelines with DataLoaders."""

    def __init__(self, map_fn=None):
        """Initialize a Runner."""
        self.map_fn = map_fn
        if map_fn is None:
            self.map_fn = lambda x, y: list(map(x, y))

    def __call__(self, pipelines_with_loaders: List[Tuple[DataLoader, Pipeline]]):
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
        return self.run(pipelines_with_loaders)

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
        results = []
        for index, (data_loader, pipeline) in enumerate(pipelines_with_loaders):
            logging.info(f"Pipeline {index + 1}/{len(pipelines_with_loaders)}...")
            results.append(self.map_fn(pipeline, data_loader))
        return results
