"""Default CLI for brain_pipe."""
import argparse
from typing import Any, Dict

from brain_pipe.cli.base import CLI
from brain_pipe.parser.base import Parser
from brain_pipe.parser.yaml import YAMLTemplateFileParser
from brain_pipe.utils.find import Finder


class DefaultCLI(CLI):
    """Default CLI for brain_pipe."""

    DEFAULT_PARSERS = ((["yaml", "yml"], YAMLTemplateFileParser()),)

    INPUT_STR = "input_file"
    PARSER_FILES_STR = "parser_files"
    PARSER_STR = "parser"
    BRAIN_PIPE_STRS = ["*", "None"]

    def get_argparser(self, description="Preprocess brain imaging data"):
        """Load the argparser.

        Returns
        -------
        argparse.ArgumentParser
            Argparser.
        """
        return argparse.ArgumentParser(description=description)

    def add_arguments_to_argparser(self, argparser: argparse.ArgumentParser):
        """Add arguments to the argparser.

        Parameters
        ----------
        argparser: argparse.ArgumentParser
            Argparser to add arguments to.

        Returns
        -------
        argparse.ArgumentParser
            Argparser with added arguments.
        """
        argparser.add_argument(
            "--parser",
            type=str,
            help="Name of the parser to use when processing the input file. If not "
            "specified, the parser will be determined based on the file extension.",
        )
        argparser.add_argument(
            f"--{self.PARSER_FILES_STR}",
            "-f",
            default=[],
            action="append",
            help="Path to external file containing the parser to use when processing "
            "the input file. Only necessary when defining your own parser. When "
            "multiple files are provided, the first parser found will be used. "
            "'None' or '*' can be used to search the files in the brain_pipe "
            "package. Default is all files in the brain_pipe package.",
        )
        argparser.add_argument(
            f"{self.INPUT_STR}",
            type=str,
            help="Input file containing the pipeline definitions.",
        )
        return argparser

    def get_parser(self, cli_info: Dict[str, Any]):
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
        if self.PARSER_STR in cli_info and cli_info[self.PARSER_STR] is not None:
            all_parsers = Finder()(Parser, cli_info[self.PARSER_FILES_STR])
            if cli_info[self.PARSER_STR] not in all_parsers:
                raise ValueError(f"Parser '{cli_info[self.PARSER_STR]}' not found.")
            return all_parsers[cli_info[self.PARSER_STR]]()

        for extensions, parser in self.DEFAULT_PARSERS:
            for extension in extensions:
                if cli_info[self.INPUT_STR].endswith(extension):
                    return parser

        raise ValueError(
            f"Could not find a parser for file '{cli_info[self.INPUT_STR]}'."
        )

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
        return parser(cli_info[self.INPUT_STR])

    def parse_cli_arguments(self, args=None):
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
        argparser = self.get_argparser()
        self.add_arguments_to_argparser(argparser)
        parsed_args_ns, _ = argparser.parse_known_args(args)
        parsed_args = vars(parsed_args_ns)
        parsed_args[self.PARSER_FILES_STR] = [
            None if x in self.BRAIN_PIPE_STRS else x
            for x in parsed_args.get(self.PARSER_FILES_STR, [])
        ]
        return parsed_args

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
        argparser = self.get_argparser(
            description=f"Required arguments for {parser.__class__.__name__}"
        )
        for required_parser_arg, argparser_args in parser.get_additional_kwargs(
            cli_info[self.INPUT_STR], cli_info
        ).items():
            if len(required_parser_arg) == 1:
                argparser.add_argument(f"-{required_parser_arg}", **argparser_args)
            else:
                argparser.add_argument(f"--{required_parser_arg}", **argparser_args)
        parsed_args_ns, _ = argparser.parse_known_args(args)
        return vars(parsed_args_ns)
