import sys
import unittest

from brain_pipe.cli import cli_entrypoint


class CLIEntrypointTest(unittest.TestCase):
    def test_help(self):
        sys.argv.append("-h")
        with self.assertRaises(SystemExit):
            cli_entrypoint()

    def test_error(self):
        sys.argv.append("blabla")
        with self.assertRaises(ValueError):
            cli_entrypoint()
