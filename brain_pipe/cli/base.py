"""Abstract base classes for command line interfaces."""

import abc
from typing import Any, Dict, List

from brain_pipe.parser.base import Parser


class CLI(abc.ABC):
    """Abstract base class for command line interfaces."""

    @abc.abstractmethod
    def parse_cli_arguments(self, args: Any = None) -> Dict[str, Any]:
        """Parse the command line arguments.

        Parameters
        ----------
        args: Any
            Arguments received from the command line. If None, the arguments should
            be parsed from sys.argv directly

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the parsed arguments.
        """
        pass

    @abc.abstractmethod
    def parse_cli_arguments_for_parser(
        self, parser: Parser, cli_info: Dict[str, Any], args: Any = None
    ) -> Dict[str, Any]:
        """Parse the command line arguments.

        Parameters
        ----------
        parser: Parser
            The parser to use to create the Pipelines and runner.
        cli_info: Dict[str, Any]
            The parsed command line arguments.
        args: Any
            Arguments received from the command line. If None, the arguments should
            be parsed from sys.argv directly

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the parsed arguments.
        """
        pass

    @abc.abstractmethod
    def get_parser(self, cli_info: Dict[str, Any]) -> Parser:
        """Find an appropriate parser based on the command line information.

        Parameters
        ----------
        cli_info: Dict[str, Any]
            The parsed command line arguments.

        Returns
        -------
        Parser
            A parser that can be used to create Pipelines and runners.
        """
        pass

    @abc.abstractmethod
    def get_run_options(self, parser: Parser, cli_info: Dict[str, Any]):
        """Extract a runner and Pipelines with their DataLoaders using the parser.

        Parameters
        ----------
        parser: Parser
            The parser to use to create the Pipelines and runner.
        cli_info: Dict[str, Any]
            The parsed command line arguments.

        Returns
        -------
        Tuple[Runner, List[Tuple[Pipeline, DataLoader]]]
            A runner and a list of tuples containing a Pipeline and its DataLoader.
        """
        pass

    def run(self, args: Any = None) -> List[List[Dict[str, Any]]]:
        """Run the command line interface.

        Parameters
        ----------
        args: Any
            Arguments received from the command line. If None, the arguments should
            be parsed from sys.argv directly

        Returns
        -------
        List[List[Dict[str, Any]]]
            A list of lists of data_dicts containing the results of the Pipelines.
        """
        cli_info = self.parse_cli_arguments(args)
        pipeline_parser = self.get_parser(cli_info)
        parser_info = self.parse_cli_arguments_for_parser(
            pipeline_parser, cli_info, args
        )
        pipeline_parser.set_additional_kwargs(parser_info)
        runner, pipelines_with_loaders = self.get_run_options(pipeline_parser, cli_info)
        return runner.run(pipelines_with_loaders)
