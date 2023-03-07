import unittest

from brain_pipe.cli import cli_factory
from brain_pipe.cli.default import DefaultCLI


class CLIFactoryTest(unittest.TestCase):
    def test_factory_default(self):
        cli = cli_factory("DefaultCLI")
        self.assertIsInstance(cli, DefaultCLI)
        cli = cli_factory("default")
        self.assertIsInstance(cli, DefaultCLI)
        cli = cli_factory(None)
        self.assertIsInstance(cli, DefaultCLI)

    def test_factory_error(self):
        with self.assertRaises(ValueError):
            cli = cli_factory("blabla")
