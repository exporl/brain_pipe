import unittest
from typing import Dict, Any, Union, Sequence, Iterator

from brain_pipe.dataloaders.base import DataLoader
from brain_pipe.pipeline.base import Pipeline, PipelineStep
from brain_pipe.runner.base import Runner


class RunnerTest(unittest.TestCase):
    class MockPipeline(Pipeline):
        def __call__(
            self, data_dict: Dict[str, Any]
        ) -> Union[Dict[str, Any], Sequence[Dict[str, Any]]]:
            return data_dict

    class MockPipelineStep(PipelineStep):
        def __call__(
            self, data_dict: Dict[str, Any]
        ) -> Union[Dict[str, Any], Sequence[Dict[str, Any]]]:
            return data_dict

    class MockRunner(Runner):
        def apply_config(self, config: Dict[str, Any]):
            return config

    class MockDataLoader(DataLoader):
        def __iter__(self) -> Iterator:
            return [{"a": 1}, {"b": {"c": [2]}}].__iter__()

    def test_run(self):
        runner = self.MockRunner()
        runner.apply_config({})
        output = runner.run(
            [
                (self.MockDataLoader(), self.MockPipeline([self.MockPipelineStep()])),
                (
                    self.MockDataLoader(),
                    self.MockPipeline(
                        [self.MockPipelineStep(), self.MockPipelineStep()]
                    ),
                ),
            ]
        )
        self.assertEqual(
            output, [[{"a": 1}, {"b": {"c": [2]}}], [{"a": 1}, {"b": {"c": [2]}}]]
        )

    def test_call(self):
        runner = self.MockRunner()
        runner.apply_config({})
        output = runner(
            [
                (self.MockDataLoader(), self.MockPipeline([self.MockPipelineStep()])),
                (
                    self.MockDataLoader(),
                    self.MockPipeline(
                        [self.MockPipelineStep(), self.MockPipelineStep()]
                    ),
                ),
            ]
        )
        self.assertEqual(
            output, [[{"a": 1}, {"b": {"c": [2]}}], [{"a": 1}, {"b": {"c": [2]}}]]
        )
