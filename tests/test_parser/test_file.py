import json
import os
import tempfile
import unittest
from typing import Union, Dict, Any, Sequence, Iterator

from brain_pipe.dataloaders.base import DataLoader
from brain_pipe.parser.file import FileLoadMixin, FileParser
from brain_pipe.parser.simple_dict import SimpleDictParser
from brain_pipe.pipeline.base import Pipeline, PipelineStep
from brain_pipe.runner.base import Runner


class MockPipelineStep(PipelineStep):
    def __init__(self, a=0):
        super().__init__()
        self.a = a

    def __call__(
        self, data_dict: Dict[str, Any]
    ) -> Union[Dict[str, Any], Sequence[Dict[str, Any]]]:
        pass


class MockPipeline(Pipeline):
    def __init__(self, steps, a=0):
        super().__init__(steps)
        self.a = a

    def __call__(
        self, data_dict: Dict[str, Any]
    ) -> Union[Dict[str, Any], Sequence[Dict[str, Any]]]:
        pass


class MockDataLoader(DataLoader):
    def __iter__(self) -> Iterator:
        pass


class MockRunner(Runner):
    def __init__(self, a=0):
        super().__init__()
        self.a = a

    def apply_config(self, config: Dict):
        pass


class FileParserTest(unittest.TestCase):
    test_data = {
        SimpleDictParser.DATALOADERS_STR: [
            {
                SimpleDictParser.CALLABLE_STR: "MockDataLoader",
                SimpleDictParser.NAME_STR: "a",
            },
            {
                SimpleDictParser.CALLABLE_STR: "MockDataLoader",
                SimpleDictParser.NAME_STR: "b",
            },
        ],
        SimpleDictParser.CONFIG_STR: {
            SimpleDictParser.RUNNER_STR: {
                SimpleDictParser.CALLABLE_STR: "MockRunner",
            }
        },
        SimpleDictParser.PIPELINES_STR: [
            {
                SimpleDictParser.CALLABLE_STR: "MockPipeline",
                "steps": [{SimpleDictParser.CALLABLE_STR: "MockPipelineStep", "a": 1}],
                SimpleDictParser.DATA_FROM_STR: "a",
            },
            {
                SimpleDictParser.CALLABLE_STR: "MockPipeline",
                "steps": [{SimpleDictParser.CALLABLE_STR: "MockPipelineStep", "a": 2}],
                SimpleDictParser.DATA_FROM_STR: "b",
            },
        ],
    }

    class MockFileParser(FileParser):
        def parse_text_to_dict(self, text: str):
            return json.loads(text)

    def test_load_file(self):
        parser = self.MockFileParser()
        parser.all_available_callables = {
            "MockDataLoader": MockDataLoader,
            "MockPipeline": MockPipeline,
            "MockPipelineStep": MockPipelineStep,
            "MockRunner": MockRunner,
        }
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as fp:
            fp.write("abcd")
            fp.seek(0)
            parsed = parser.load_file(fp.name)
        self.assertEqual(parsed, "abcd")

    def test_call(self):
        parser = self.MockFileParser()
        parser.all_available_callables = {
            "MockDataLoader": MockDataLoader,
            "MockPipeline": MockPipeline,
            "MockPipelineStep": MockPipelineStep,
            "MockRunner": MockRunner,
        }
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as fp:
            json.dump(self.test_data, fp)
            fp.seek(0)
            parsed = parser(fp.name)
        self.assertIsInstance(parsed[0], MockRunner)
        self.assertIsInstance(parsed[1][0][0], MockDataLoader)
        self.assertIsInstance(parsed[1][1][0], MockDataLoader)
        self.assertNotEqual(parsed[1][1][0], parsed[1][0][0])
        self.assertIsInstance(parsed[1][0][1], MockPipeline)
        self.assertEqual(parsed[1][0][1].steps[0].a, 1)
        self.assertEqual(parsed[1][1][1].steps[0].a, 2)
