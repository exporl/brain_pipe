import unittest
from typing import Dict, Any, Union, Sequence

from brain_pipe.dataloaders.base import DataLoader
from brain_pipe.parser.base import Parser
from brain_pipe.pipeline.base import Pipeline
from brain_pipe.runner.base import Runner


class ParserTest(unittest.TestCase):
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
        def __call__(self):
            return ParserTest.MockRunner(), [
                (ParserTest.MockDataLoader(), ParserTest.MockPipeline([]))
            ]

    def test_call(self):
        runner, pipelines_with_dataloaders = self.MockParser()()
        self.assertIsInstance(runner, self.MockRunner)
        self.assertEqual(len(pipelines_with_dataloaders), 1)
        self.assertIsInstance(pipelines_with_dataloaders[0][0], self.MockDataLoader)
        self.assertIsInstance(pipelines_with_dataloaders[0][1], self.MockPipeline)
        output = list(
            map(pipelines_with_dataloaders[0][1], pipelines_with_dataloaders[0][0])
        )
        self.assertEqual(output, [{"a": 1}, {"b": 2}])

    def test_link_loader_to_pipeline_multiple(self):
        parser = self.MockParser()
        pipelines = (
            ("a", self.MockPipeline([])),
            ("b", self.MockPipeline([])),
            ("a", self.MockPipeline([])),
        )
        data_loaders = {
            "a": self.MockDataLoader(),
            "b": self.MockDataLoader(),
        }
        matched = parser.link_loader_to_pipeline(pipelines, data_loaders)
        self.assertEqual(len(matched), 3)
        self.assertEqual(matched[0][0], data_loaders["a"])
        self.assertEqual(matched[0][1], pipelines[0][1])
        self.assertEqual(matched[1][0], data_loaders["b"])
        self.assertEqual(matched[1][1], pipelines[1][1])
        self.assertEqual(matched[2][0], data_loaders["a"])
        self.assertEqual(matched[2][1], pipelines[2][1])

    def test_link_loader_to_pipeline_no_loader(self):
        parser = self.MockParser()
        pipelines = (
            ("a", self.MockPipeline([])),
            ("b", self.MockPipeline([])),
            ("a", self.MockPipeline([])),
        )
        data_loaders = {}
        with self.assertRaises(ValueError):
            parser.link_loader_to_pipeline(pipelines[0], data_loaders)
