"""Module to define command line interfaces."""
import sys
from typing import Optional
from brain_pipe.utils.log import default_logging

from brain_pipe.cli.base import CLI


def cli_factory(cli_name: Optional[str]) -> CLI:
    """Create a CLI object based on the command line arguments.

    Parameters
    ----------
    cli_name: Optional[str]
        The name of the CLI to create.

    Returns
    -------
    CLI
        A CLI object that can be used to create Pipelines

    Raises
    ------
    ValueError
        If the CLI is not found.
    """
    if cli_name in ["DefaultCLI", "default", None]:
        from brain_pipe.cli.default import DefaultCLI

        return DefaultCLI()
    else:
        raise ValueError(f"CLI '{cli_name}' not found.")


def cli_entrypoint():
    """Run the command line interface."""
    cli = cli_factory("DefaultCLI")
    # Check if the user has specified a CLI
    if len(sys.argv) > 1:
        try:
            cli = cli_factory(sys.argv[1])
            sys.argv.pop(1)
        except ValueError:
            # If the CLI is not found, use the default
            pass
    default_logging()
    cli.run()
