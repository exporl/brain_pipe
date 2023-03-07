import argparse
import os.path
import sys
import unittest

from brain_pipe.cli.default import DefaultCLI
from brain_pipe.parser.base import Parser


class MockParser(Parser):
    def __call__(self, input_):
        return input_


class DefaultCLITest(unittest.TestCase):
    def setUp(self) -> None:
        self.cli = DefaultCLI()
        self.default_parsers = self.cli.DEFAULT_PARSERS

    def test_get_argparser(self):
        parser = self.cli.get_argparser()
        self.assertIsInstance(parser, argparse.ArgumentParser)
        self.assertEqual(parser.description, "Preprocess brain imaging data")

    def test_add_arguments_to_argparser(self):
        parser = argparse.ArgumentParser()
        self.cli.add_arguments_to_argparser(parser)
        self.assertEqual(len(parser._actions), 4)

    def test_get_parser_no_info(self):
        with self.assertRaises(ValueError):
            self.cli.get_parser({"input_file": "test.123"})

    def test_get_parser_parser_unknown(self):
        with self.assertRaises(ValueError):
            self.cli.get_parser({"parser": "unkown", "parser_files": []})

    def test_get_parser_parser_known(self):
        parser = self.cli.get_parser(
            {"parser": "MockParser", "parser_files": [os.path.abspath(__file__)]}
        )
        # Normal equality doesn't work
        self.assertEqual(parser.__class__.__name__, MockParser.__name__)

    def test_get_parser_default_parsers(self):
        self.cli.DEFAULT_PARSERS = ((["123"], MockParser()),)
        parser = self.cli.get_parser(
            {"parser_files": [os.path.abspath(__file__)], "input_file": "test.123"}
        )
        self.assertIsInstance(parser, MockParser)

    def test_get_run_options(self):
        output = self.cli.get_run_options(MockParser(), {"input_file": "test.123"})
        self.assertEqual(output, "test.123")

    def test_parse_cli_arguments_api(self):
        raw_args = ["--parser", "MockParser", "-f", "abc.def", "-f", "test.123", "a.b"]
        parsed_args = self.cli.parse_cli_arguments(raw_args)
        self.assertEqual(
            parsed_args,
            {
                "input_file": "a.b",
                "parser": "MockParser",
                "parser_files": ["abc.def", "test.123"],
            },
        )

    def test_parse_cli_arguments_sysargv(self):
        raw_args = ["--parser", "MockParser", "-f", "abc.def", "-f", "test.123", "a.b"]
        # Manually adapt sys.argv
        sys.argv[1:] = raw_args
        parsed_args = self.cli.parse_cli_arguments()
        self.assertEqual(
            parsed_args,
            {
                "input_file": "a.b",
                "parser": "MockParser",
                "parser_files": ["abc.def", "test.123"],
            },
        )

    def test_parse_cli_arguments_parser_files(self):
        raw_args = ["--parser", "MockParser", "-f", "*", "-f", "None", "a.b"]
        parsed_args = self.cli.parse_cli_arguments(raw_args)
        self.assertEqual(
            parsed_args,
            {"input_file": "a.b", "parser": "MockParser", "parser_files": [None, None]},
        )

    def tearDown(self) -> None:
        self.cli.DEFAULT_PARSERS = self.default_parsers
