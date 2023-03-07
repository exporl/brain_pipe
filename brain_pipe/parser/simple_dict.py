"""Parsers that can parse dictionaries contain only text."""
import importlib
import inspect
from typing import Dict, Sequence, Any

from brain_pipe.dataloaders.base import DataLoader
from brain_pipe.parser.base import Parser
from brain_pipe.pipeline.base import Pipeline
from brain_pipe.runner.default import DefaultRunner
from brain_pipe.utils import ObjectsWithConfig
from brain_pipe.utils.find import Finder


class SimpleDictParser(Parser, ObjectsWithConfig):
    """Parser that can parse dictionaries contain only text."""

    CALLABLE_STR = "callable"
    POINTER_STR = "is_pointer"
    DATA_FROM_STR = "data_from"
    NAME_STR = "name"
    CONFIG_STR = "config"
    PARSER_STR = "parser"
    EXTRA_PATHS_STR = "extra_paths"
    PIPELINES_STR = "pipelines"
    DATALOADERS_STR = "dataloaders"
    RUNNER_STR = "runner"
    RUNNER_CLASS_STR = "class"
    ARGS_STR = "*args"

    def __init__(self):
        """Initialize the parser."""
        self.finder = Finder()
        self.all_available_callables = {}

    def __call__(self, text_dict: Dict[str, Any]):
        """Parse the text dictionary.

        Parameters
        ----------
        text_dict: Dict[str, Any]
            The text dictionary to parse.

        Returns
        -------
        Tuple[Runner, List[Tuple[Pipeline, DataLoader]]]
            A runner and a list of tuples containing a Pipeline and its DataLoader.
        """
        parser_config = {}
        if (
            self.CONFIG_STR in text_dict
            and self.PARSER_STR in text_dict[self.CONFIG_STR]  # noqa: W503
        ):
            parser_config = text_dict[self.CONFIG_STR][self.PARSER_STR]
        self.apply_config(parser_config)
        return self.parse_all(text_dict)

    def get_additional_kwargs(
        self, input_: Any, info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Find the additional arguments that should be passed to the parser.

        This function identifies additional arguments that should be passed to the
        parser by e.g. a CLI.

        Parameters
        ----------
        input_: Dict[str, Any]
            The input for the parser.

        info: Dict[str, Any]
            A dictionary containing available info, such as e.g. parsed CLI arguments.
            This information can be used to determine the additional arguments.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the names of the arguments as keys and the options
            (kwargs) for a (CLI) argument parser as values.
        """
        return {
            self.EXTRA_PATHS_STR: {
                "help": "Paths to files containing custom callables.",
                "nargs": "+",
                "default": [],
                "required": False,
            }
        }

    def set_additional_kwargs(self, args: Dict[str, Any]):
        """Set the additional arguments defined in :meth:`get_additional_kwargs`.

        Parameters
        ----------
        args: Dict[str, Any]
            The additional arguments.

        Notes
        -----
        This method calls :meth:`apply_config` internally, so the additional arguments
        can also be specified in the dictionary itself under the parser configuration.
        """
        self.apply_config({self.EXTRA_PATHS_STR: args.get(self.EXTRA_PATHS_STR, [])})

    def apply_config(self, config: Dict[str, Any]):
        """Apply the configuration to the parser.

        Parameters
        ----------
        config: Dict[str, Any]
            The configuration to apply.
        """
        extra_paths = config.get(self.EXTRA_PATHS_STR, [])
        self.all_available_callables.update(
            self.finder(None, [None] + extra_paths, True)
        )

    def get_runner_from_parsed_config(self, parsed_config: Dict[str, Any]):
        """Construct a runner from the configuration.

        Parameters
        ----------
        parsed_config: Dict[str, Any]
            The already parsed configuration.

        Returns
        -------
        Runner
        """
        runner = parsed_config.get(self.RUNNER_STR, DefaultRunner())
        return runner

    def parse_all(self, all_info: Dict[str, Any]):
        """Parse all information in the text dictionary.

        Parameters
        ----------
        all_info: Dict[str, Any]
            The text dictionary to parse.

        Returns
        -------
        Tuple[Runner, List[Tuple[Pipeline, DataLoader]]]
            A runner and a list of tuples containing a Pipeline and its DataLoader.
        """
        parsed_config = {}
        pipelines = {}
        loaders = {}
        for key, value in all_info.items():
            if key == self.CONFIG_STR:
                parsed_config = self.recursive_parse(value)
            elif key == self.PIPELINES_STR:
                pipelines = []
                if not isinstance(value, Sequence):
                    value = [value]
                for pipeline_dict in value:
                    result = self.recursive_parse(pipeline_dict)
                    if not isinstance(result, tuple):
                        raise ValueError(f"Pipeline has no '{self.DATA_FROM_STR}' key.")
                    pipelines.append(result)
            elif key == self.DATALOADERS_STR:
                if not isinstance(value, Sequence):
                    value = [value]
                loaders = dict([self.recursive_parse(x) for x in value])
            else:
                raise ValueError(f"'{key}' is not a valid option.")

        runner = self.get_runner_from_parsed_config(parsed_config)
        return runner, self.link_loader_to_pipeline(pipelines, loaders)

    def recursive_parse(self, text_dict: Dict[str, Any]):
        """Parse the configuration.

        Parameters
        ----------
        text_dict: Dict[str, Any]
            The text dictionary to parse.

        Returns
        -------
        Union[Dict[str, Any], Tuple[str, Dict[str, Any]]]
            The parsed dictionary (i.e. with objects).
        """
        if self.CALLABLE_STR in text_dict:
            return self._object_constructor(text_dict)

        for key, value in list(text_dict.items()):
            if isinstance(value, dict):
                text_dict[key] = self.recursive_parse(value)

            elif isinstance(value, (list, tuple)):
                new_list = []
                for item in value:
                    if isinstance(item, dict):
                        new_list.append(self.recursive_parse(item))
                    else:
                        new_list.append(item)
                text_dict[key] = new_list
        return text_dict

    def _dynamic_import(self, cls_name: str):
        cls = None
        try:
            split_name = cls_name.split(".")
            module_name = ".".join(split_name[:-1])
            new_cls_name = split_name[-1]
            cls = getattr(importlib.import_module(module_name), new_cls_name)
        except (ImportError, AttributeError):
            pass
        return cls

    def _object_constructor(self, text_dict: Dict[str, Any]):
        callable_name = text_dict.pop(self.CALLABLE_STR)
        # Get the implementation of callable based on the name of the callable
        callabl = None
        if callable_name in self.all_available_callables:
            callabl = self.all_available_callables[callable_name]
        # Check if we can import it dynamically
        elif "." in callable_name:
            callabl = self._dynamic_import(callable_name)
        if callabl is None:
            raise ValueError(
                f"Unknown callable '{callable_name}', "
                f"did you forget to import it from a custom file?"
            )

        # Check if only a pointer to a callable should be produced
        if self.POINTER_STR in text_dict:
            return callabl

        # Check for special classes
        prefix = None
        # Check if a pipeline
        if inspect.isclass(callabl) and issubclass(callabl, Pipeline):
            prefix = text_dict.pop(self.DATA_FROM_STR, None)
        # Check if a DataLoader
        elif inspect.isclass(callabl) and issubclass(callabl, DataLoader):
            if self.NAME_STR not in text_dict:
                raise ValueError(
                    f"A DataLoader must have a name (with key '{self.NAME_STR}'), "
                    f"didn't find one in {text_dict}"
                )
            prefix = text_dict.pop(self.NAME_STR)

        # Parse the arguments
        kwargs = self.recursive_parse(text_dict)
        # Support for *args
        args = kwargs.pop(self.ARGS_STR, [])

        # CLASS_STR is already removed from the text_dict
        obj = callabl(*args, **kwargs)
        if prefix is not None:
            return prefix, obj
        return obj
