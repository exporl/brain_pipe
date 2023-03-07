"""Base parser class to convert information into pipelines, loaders and runners."""
import abc
from typing import Dict, Sequence, Tuple, Any

from brain_pipe.dataloaders.base import DataLoader
from brain_pipe.pipeline.base import Pipeline


class Parser(abc.ABC):
    """Base parser class to convert information into pipelines, loaders and runners."""

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        """Parse the information.

        Parameters
        ----------
        args: Any
            Positional arguments.
        kwargs: Any
            Keyword arguments.

        Returns
        -------
        Tuple[Runner, List[Tuple[Pipeline, DataLoader]]]
            A runner and a list of tuples containing a Pipeline and its DataLoader.
        """
        pass

    def link_loader_to_pipeline(
        self, pipelines: Sequence[Tuple[str, Pipeline]], loaders: Dict[str, DataLoader]
    ):
        """Pair pipelines with their corresponding dataloaders.

        Parameters
        ----------
        pipelines: Sequence[Tuple[str, Pipeline]]
            A dictionary containing a mapping between the name of a loader and
            the Pipeline object
        loaders: Dict[str, DataLoader]
            A dictionary containing a mapping between the name of a loader and
            the DataLoader object

        Returns
        -------
        List[Tuple[DataLoader, Pipeline]]
            A list of tuples containing a DataLoader and its corresponding Pipeline.
        """
        merged = []
        for loader_name, pipeline in pipelines:
            if loader_name not in loaders:
                raise ValueError(
                    f"Pipeline '{pipeline}' expected a DataLoader with name "
                    f"'{loader_name}' but it is not present. Available loaders are: "
                    f"{','.join(loaders.keys())}"
                )

            merged.append((loaders[loader_name], pipeline))
        return merged

    def get_additional_kwargs(
        self, input_: Any, info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Find the additional arguments that should be passed to the parser.

        This function identifies additional arguments that should be passed to the
        parser by e.g. a CLI.

        Parameters
        ----------
        input_: Any
            The input that should be parsed.
        info: Dict[str, Any]
            A dictionary containing available info, such as e.g. parsed CLI arguments.
            This information can be used to determine the additional arguments.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the names of the arguments as keys and the options
            (kwargs) for a (CLI) argument parser as values.

        """
        return {}

    def set_additional_kwargs(self, args: Dict[str, Any]):
        """Set additional arguments for the parser.

        Parameters
        ----------
        args: Dict[str, Any]
            A dictionary containing the names of the arguments as keys and their values
            as values.
        """
        for key, value in args.items():
            setattr(self, key, value)
