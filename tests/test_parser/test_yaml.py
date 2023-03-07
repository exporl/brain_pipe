import os
import tempfile
import unittest
from typing import Dict, Any, Sequence, Union, Iterator

from brain_pipe.dataloaders.base import DataLoader
from brain_pipe.parser.yaml import (
    YAMLTextParser,
    YAMLFileParser,
    YAMLTemplateFileParser,
)
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
        self.a = a

    def apply_config(self, config: Dict):
        pass


class YAMLParserTest(unittest.TestCase):
    def test_yaml_file_parser(self):
        parser = YAMLFileParser()
        test_data_folder = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_data"
        )
        test_yaml_template = os.path.join(test_data_folder, "test.yaml")

        with open(test_yaml_template, "r") as fp:
            yaml_template_text = fp.read()

        # Insert the path to this file into the yaml template
        new_yaml = yaml_template_text.replace(
            "{{extra_path}}", os.path.abspath(__file__)
        )

        with tempfile.NamedTemporaryFile("w+") as fp:
            fp.write(new_yaml)
            fp.seek(0)
            name = fp.name
            parsed = parser(name)
        self.assertEqual(parsed[0].__class__.__name__, "MockRunner")
        self.assertEqual(parsed[1][0][0].__class__.__name__, "MockDataLoader")
        self.assertEqual(parsed[1][1][0].__class__.__name__, "MockDataLoader")
        self.assertNotEqual(parsed[1][1][0], parsed[1][0][0])
        self.assertEqual(parsed[1][0][1].__class__.__name__, "MockPipeline")
        self.assertEqual(parsed[1][0][1].steps[0].a, 1)
        self.assertEqual(parsed[1][1][1].steps[0].a, 2)

    def test_yaml_template_file_parser(self):
        parser = YAMLTemplateFileParser()
        test_data_folder = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_data"
        )
        test_yaml_template = os.path.join(test_data_folder, "test.yaml")

        parser.set_additional_kwargs({"extra_path": os.path.abspath(__file__)})
        parsed = parser(test_yaml_template)
        self.assertEqual(parsed[0].__class__.__name__, "MockRunner")
        self.assertEqual(parsed[1][0][0].__class__.__name__, "MockDataLoader")
        self.assertEqual(parsed[1][1][0].__class__.__name__, "MockDataLoader")
        self.assertNotEqual(parsed[1][1][0], parsed[1][0][0])
        self.assertEqual(parsed[1][0][1].__class__.__name__, "MockPipeline")
        self.assertEqual(parsed[1][0][1].steps[0].a, 1)
        self.assertEqual(parsed[1][1][1].steps[0].a, 2)

    def test_yaml_text_parser(self):
        parser = YAMLTextParser()
        test_data_folder = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_data"
        )
        test_yaml_template = os.path.join(test_data_folder, "test.yaml")

        with open(test_yaml_template, "r") as fp:
            yaml_template_text = fp.read()

        # Insert the path to this file into the yaml template
        new_yaml = yaml_template_text.replace(
            "{{extra_path}}", os.path.abspath(__file__)
        )

        parsed = parser(new_yaml)
        self.assertEqual(parsed[0].__class__.__name__, "MockRunner")
        self.assertEqual(parsed[1][0][0].__class__.__name__, "MockDataLoader")
        self.assertEqual(parsed[1][1][0].__class__.__name__, "MockDataLoader")
        self.assertNotEqual(parsed[1][1][0], parsed[1][0][0])
        self.assertEqual(parsed[1][0][1].__class__.__name__, "MockPipeline")
        self.assertEqual(parsed[1][0][1].steps[0].a, 1)
        self.assertEqual(parsed[1][1][1].steps[0].a, 2)
