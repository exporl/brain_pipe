import unittest
from typing import Dict, Any, Union, Sequence

from brain_pipe.cli.base import CLI
from brain_pipe.dataloaders.base import DataLoader
from brain_pipe.parser.base import Parser
from brain_pipe.pipeline.base import Pipeline
from brain_pipe.runner.base import Runner


class CLITest(unittest.TestCase):
    class MockDataLoader(DataLoader):
        def __iter__(self):
            return [{"a": 1}, {"b": 2}].__iter__()

    class MockPipeline(Pipeline):
        def __call__(
            self, data_dict: Dict[str, Any]
        ) -> Union[Dict[str, Any], Sequence[Dict[str, Any]]]:
            return data_dict

    class MockRunner(Runner):
        def apply_config(self, config: Dict):
            pass

    class MockParser(Parser):
        def __call__(self, *args, **kwargs):
            return CLITest.MockRunner()

    class MockCLI(CLI):
        def parse_cli_arguments_for_parser(
            self, parser: Parser, cli_info: Dict[str, Any], args: Any = None
        ) -> Dict[str, Any]:
            return cli_info

        def parse_cli_arguments(self, args=None):
            return {"a": 1}

        def get_parser(self, cli_info):
            return CLITest.MockParser()

        def get_run_options(self, parser, cli_info):
            pipelines_with_data_loaders = [
                (
                    CLITest.MockDataLoader(),
                    CLITest.MockPipeline([]),
                ),
                (CLITest.MockDataLoader(), CLITest.MockPipeline([])),
            ]
            return parser(), pipelines_with_data_loaders

    def test_run(self):
        cli = CLITest.MockCLI()
        self.assertEqual(cli.run(), [[{"a": 1}, {"b": 2}], [{"a": 1}, {"b": 2}]])

    def test_parse_cli_arguments_for_parser(self):
        cli = CLITest.MockCLI()
        self.assertEqual(
            cli.parse_cli_arguments_for_parser(None, {"a": 1}, None), {"a": 1}
        )
