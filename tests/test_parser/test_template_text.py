import json
import unittest
from typing import Dict, Any, Sequence, Union, Iterator

import jinja2

from brain_pipe.dataloaders.base import DataLoader
from brain_pipe.parser.simple_dict import SimpleDictParser
from brain_pipe.parser.template_text import TemplateTextParser
from brain_pipe.pipeline.base import PipelineStep, Pipeline
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


class TemplateTextParserTest(unittest.TestCase):
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
                "steps": [
                    {SimpleDictParser.CALLABLE_STR: "MockPipelineStep", "a": 123}
                ],
                SimpleDictParser.DATA_FROM_STR: "b",
            },
        ],
    }

    class MockTemplateTextParser(TemplateTextParser):
        def parse_text_to_dict(self, text):
            return json.loads(text)

    def test_get_additional_args_empty(self):
        parser = self.MockTemplateTextParser()
        args = parser.get_additional_kwargs('{"a": 1, "b": 2}', {})
        self.assertEqual(
            args,
            {
                "extra_paths": {
                    "help": "Paths to files containing custom callables.",
                    "nargs": "+",
                    "default": [],
                    "required": False,
                }
            },
        )

    def test_get_additional_args_undeclared(self):
        parser = self.MockTemplateTextParser()
        args = parser.get_additional_kwargs('{"a": {{ c }}, "b": {{ d }}}', {})
        self.assertEqual(
            args,
            {
                "extra_paths": {
                    "help": "Paths to files containing custom callables.",
                    "nargs": "+",
                    "default": [],
                    "required": False,
                },
                "d": {
                    "help": "Variable 'd', as used in the template",
                    "required": True,
                },
                "c": {
                    "help": "Variable 'c', as used in the template",
                    "required": True,
                },
            },
        )

    def test_set_additional_args(self):
        parser = self.MockTemplateTextParser()
        parser.set_additional_kwargs({"a": 1, "b": 2})
        self.assertEqual(parser.environment.globals["a"], 1)
        self.assertEqual(parser.environment.globals["b"], 2)

    def test_call_no_template(self):
        parser = self.MockTemplateTextParser()
        parser.all_available_callables = {
            "MockDataLoader": MockDataLoader,
            "MockPipeline": MockPipeline,
            "MockPipelineStep": MockPipelineStep,
            "MockRunner": MockRunner,
        }

        parsed = parser(json.dumps(TemplateTextParserTest.test_data))

        self.assertIsInstance(parsed[0], MockRunner)
        self.assertIsInstance(parsed[1][0][0], MockDataLoader)
        self.assertIsInstance(parsed[1][1][0], MockDataLoader)
        self.assertNotEqual(parsed[1][1][0], parsed[1][0][0])
        self.assertIsInstance(parsed[1][0][1], MockPipeline)
        self.assertEqual(parsed[1][0][1].steps[0].a, 1)
        self.assertEqual(parsed[1][1][1].steps[0].a, 123)

    def test_call(self):
        parser = self.MockTemplateTextParser()
        parser.all_available_callables = {
            "MockDataLoader": MockDataLoader,
            "MockPipeline": MockPipeline,
            "MockPipelineStep": MockPipelineStep,
            "MockRunner": MockRunner,
        }

        parser.environment = jinja2.Environment(**parser.DEFAULT_ENVIRONMENT_KWARGS)
        parser.environment.globals["number"] = 456
        parsed = parser(
            json.dumps(TemplateTextParserTest.test_data).replace("123", "{{ number }}")
        )

        self.assertIsInstance(parsed[0], MockRunner)
        self.assertIsInstance(parsed[1][0][0], MockDataLoader)
        self.assertIsInstance(parsed[1][1][0], MockDataLoader)
        self.assertNotEqual(parsed[1][1][0], parsed[1][0][0])
        self.assertIsInstance(parsed[1][0][1], MockPipeline)
        self.assertEqual(parsed[1][0][1].steps[0].a, 1)
        self.assertEqual(parsed[1][1][1].steps[0].a, 456)
