import os.path
import unittest
from typing import Dict, Any, Union, Sequence, Iterator

import numpy as np

from brain_pipe.dataloaders.base import DataLoader
from brain_pipe.parser.simple_dict import SimpleDictParser
from brain_pipe.pipeline.base import Pipeline, PipelineStep
from brain_pipe.runner.base import Runner
from brain_pipe.runner.default import DefaultRunner


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


class SimpleDictParserTest(unittest.TestCase):
    class MockClassA:
        def __init__(self, a, b=2):
            self.a = a
            self.b = b

        def __eq__(self, other):
            return (
                self.a == other.a
                and self.b == other.b
                and isinstance(other, SimpleDictParserTest.MockClassA)
            )

    class MockClassB:
        def __eq__(self, other):
            return isinstance(other, SimpleDictParserTest.MockClassB)

    def test_apply_config_empty(self):
        parser = SimpleDictParser()
        parser.apply_config({})

        self.assertIn("DefaultPipeline", parser.all_available_callables)
        self.assertIn("CachingPreprocessingPipeline", parser.all_available_callables)
        self.assertNotIn("MockPipeline", parser.all_available_callables)

        self.assertIn("DataLoader", parser.all_available_callables)
        self.assertIn("GlobLoader", parser.all_available_callables)
        self.assertNotIn("MockDataLoader", parser.all_available_callables)

        self.assertIn("AlignPeriodicBlockTriggers", parser.all_available_callables)
        self.assertNotIn("MockPipeline", parser.all_available_callables)

    def test_apply_config_extra_paths(self):
        parser = SimpleDictParser()
        parser.apply_config(
            {SimpleDictParser.EXTRA_PATHS_STR: [os.path.abspath(__file__)]}
        )
        self._test_extra_paths_classes(parser)

    def test_apply_config_extra_paths_cwd(self):
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        filename = os.path.basename(__file__)
        parser = SimpleDictParser()
        parser.apply_config(
            {SimpleDictParser.EXTRA_PATHS_STR: [os.path.join(".", filename)]}
        )
        self._test_extra_paths_classes(parser)

    def _test_extra_paths_classes(self, parser):
        self.assertIn("DefaultPipeline", parser.all_available_callables)
        self.assertIn("CachingPreprocessingPipeline", parser.all_available_callables)
        self.assertIn("MockPipeline", parser.all_available_callables)

        self.assertIn("DataLoader", parser.all_available_callables)
        self.assertIn("GlobLoader", parser.all_available_callables)
        self.assertIn("MockDataLoader", parser.all_available_callables)

        self.assertIn("AlignPeriodicBlockTriggers", parser.all_available_callables)
        self.assertIn("MockPipeline", parser.all_available_callables)

    def test_get_runner_from_config_empty(self):
        parser = SimpleDictParser()
        runner = parser.get_runner_from_parsed_config({})
        self.assertIsInstance(runner, DefaultRunner)

    def test_get_runner_from_config_mock(self):
        parser = SimpleDictParser()
        runner = parser.get_runner_from_parsed_config(
            {SimpleDictParser.RUNNER_STR: MockRunner()}
        )
        self.assertIsInstance(runner, MockRunner)

    def test_parse_config(self):
        parser = SimpleDictParser()
        parser.all_available_callables = {
            "MockClassA": self.MockClassA,
            "MockClassB": self.MockClassB,
        }
        parsed = parser.recursive_parse(
            {
                "a": 1,
                "b": {SimpleDictParser.CALLABLE_STR: "MockClassA", "a": 2},
                "c": [{SimpleDictParser.CALLABLE_STR: "MockClassB"}, 3],
                "d": {SimpleDictParser.CALLABLE_STR: "MockClassB"},
                "e": {
                    "a": {
                        "a": {
                            SimpleDictParser.CALLABLE_STR: "MockClassA",
                            "a": 2,
                            "b": -1,
                        }
                    },
                    "b": 5,
                },
            }
        )
        self.assertEqual(
            parsed,
            {
                "a": 1,
                "b": self.MockClassA(2, b=2),
                "c": [self.MockClassB(), 3],
                "d": self.MockClassB(),
                "e": {"a": {"a": self.MockClassA(a=2, b=-1)}, "b": 5},
            },
        )

    def test_recursive_parse_single(self):
        parser = SimpleDictParser()
        parser.all_available_callables = {"MockClassB": self.MockClassB}
        parsed = parser.recursive_parse({SimpleDictParser.CALLABLE_STR: "MockClassB"})
        self.assertEqual(parsed, self.MockClassB())

    def test_object_constructor_pipeline(self):
        parser = SimpleDictParser()
        parser.all_available_callables = {"Pipeline": MockPipeline}
        obj = parser._object_constructor(
            {SimpleDictParser.CALLABLE_STR: "Pipeline", "steps": []}
        )
        self.assertIsInstance(obj, MockPipeline)
        self.assertEqual(obj.steps, [])

        obj = parser._object_constructor(
            {
                SimpleDictParser.CALLABLE_STR: "Pipeline",
                SimpleDictParser.DATA_FROM_STR: "a",
                "steps": [],
            }
        )
        self.assertEqual(obj[0], "a")
        self.assertIsInstance(obj[1], MockPipeline)
        self.assertEqual(obj[1].steps, [])

    def test_object_constructor_dataloader(self):
        parser = SimpleDictParser()
        parser.all_available_callables = {"DataLoader": MockDataLoader}
        with self.assertRaises(ValueError):
            parser._object_constructor({SimpleDictParser.CALLABLE_STR: "DataLoader"})
        obj = parser._object_constructor(
            {
                SimpleDictParser.CALLABLE_STR: "DataLoader",
                SimpleDictParser.NAME_STR: "a",
            }
        )
        self.assertEqual(obj[0], "a")
        self.assertIsInstance(obj[1], MockDataLoader)

    def test_object_constructor_dynamic_class(self):
        parser = SimpleDictParser()
        parser.all_available_callables = {}
        parsed = parser._object_constructor(
            {
                SimpleDictParser.CALLABLE_STR: "scipy.signal.butter",
                "N": 1,
                "Wn": 0.5,
                "btype": "highpass",
                "fs": 1024,
                "output": "sos",
            }
        )
        self.assertTrue(
            np.isclose(
                parsed,
                [
                    [
                        0.9984683675055748,
                        -0.9984683675055748,
                        0.0,
                        1.0,
                        -0.9969367350111495,
                        0.0,
                    ]
                ],
            ).all()
        )

    def test_object_constructor_pointer(self):
        parser = SimpleDictParser()
        parser.all_available_callables = {"MockClassB": self.MockClassB}
        parsed = parser._object_constructor(
            {
                SimpleDictParser.CALLABLE_STR: "MockClassB",
                SimpleDictParser.POINTER_STR: True,
            }
        )
        self.assertEqual(parsed, self.MockClassB)

    def test_object_constructor_func(self):
        parser = SimpleDictParser()

        def _temp_fn(a, b):
            return a + b

        parser.all_available_callables = {"_temp_fn": _temp_fn}
        parsed = parser._object_constructor(
            {SimpleDictParser.CALLABLE_STR: "_temp_fn", "a": 1, "b": 2}
        )
        self.assertEqual(parsed, 3)

    def test_object_constructor_args(self):
        parser = SimpleDictParser()

        def test_fn(a, b, *args, **kwargs):
            return a, b, args, kwargs

        parser.all_available_callables = {"test_fn": test_fn}
        parsed = parser._object_constructor(
            {
                SimpleDictParser.CALLABLE_STR: "test_fn",
                SimpleDictParser.ARGS_STR: [1, 2, 3],
                "d": 3,
                "e": 4,
            }
        )
        self.assertEqual(parsed, (1, 2, (3,), {"d": 3, "e": 4}))

    def test_object_constructor_no_dynamic_found(self):
        parser = SimpleDictParser()
        parser.all_available_callables = {}
        with self.assertRaises(ValueError):
            parser._object_constructor(
                {
                    SimpleDictParser.CALLABLE_STR: "scipy.signal.thisdoesnotexists",
                    "N": 1,
                }
            )

    def test_object_constructor_error(self):
        parser = SimpleDictParser()
        with self.assertRaises(ValueError):
            parser._object_constructor({SimpleDictParser.CALLABLE_STR: "a"})

    def test_parse_all(self):
        parser = SimpleDictParser()
        parser.all_available_callables = {
            "DataLoader": MockDataLoader,
            "Pipeline": MockPipeline,
            "PipelineStep": MockPipelineStep,
            SimpleDictParser.RUNNER_STR: MockRunner,
        }

        parsed = parser.parse_all(
            {
                SimpleDictParser.DATALOADERS_STR: [
                    {
                        SimpleDictParser.CALLABLE_STR: "DataLoader",
                        SimpleDictParser.NAME_STR: "a",
                    },
                    {
                        SimpleDictParser.CALLABLE_STR: "DataLoader",
                        SimpleDictParser.NAME_STR: "b",
                    },
                ],
                SimpleDictParser.CONFIG_STR: {
                    SimpleDictParser.RUNNER_STR: {
                        SimpleDictParser.CALLABLE_STR: SimpleDictParser.RUNNER_STR,
                    }
                },
                SimpleDictParser.PIPELINES_STR: [
                    {
                        SimpleDictParser.CALLABLE_STR: "Pipeline",
                        "steps": [
                            {SimpleDictParser.CALLABLE_STR: "PipelineStep", "a": 1}
                        ],
                        SimpleDictParser.DATA_FROM_STR: "a",
                    },
                    {
                        SimpleDictParser.CALLABLE_STR: "Pipeline",
                        "steps": [
                            {SimpleDictParser.CALLABLE_STR: "PipelineStep", "a": 2}
                        ],
                        SimpleDictParser.DATA_FROM_STR: "b",
                    },
                ],
            }
        )

        self.assertIsInstance(parsed[0], MockRunner)
        self.assertIsInstance(parsed[1][0][0], MockDataLoader)
        self.assertIsInstance(parsed[1][1][0], MockDataLoader)
        self.assertNotEqual(parsed[1][1][0], parsed[1][0][0])
        self.assertIsInstance(parsed[1][0][1], MockPipeline)
        self.assertEqual(parsed[1][0][1].steps[0].a, 1)
        self.assertEqual(parsed[1][1][1].steps[0].a, 2)

    def test_call(self):
        parser = SimpleDictParser()
        parsed = parser(
            {
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
                    },
                    SimpleDictParser.PARSER_STR: {
                        SimpleDictParser.EXTRA_PATHS_STR: [__file__]
                    },
                },
                SimpleDictParser.PIPELINES_STR: [
                    {
                        SimpleDictParser.CALLABLE_STR: "MockPipeline",
                        "steps": [
                            {SimpleDictParser.CALLABLE_STR: "MockPipelineStep", "a": 1}
                        ],
                        SimpleDictParser.DATA_FROM_STR: "a",
                    },
                    {
                        SimpleDictParser.CALLABLE_STR: "MockPipeline",
                        "steps": [
                            {SimpleDictParser.CALLABLE_STR: "MockPipelineStep", "a": 2}
                        ],
                        SimpleDictParser.DATA_FROM_STR: "b",
                    },
                ],
            }
        )

        self.assertEqual(parsed[0].__class__.__name__, "MockRunner")
        self.assertEqual(parsed[1][0][0].__class__.__name__, "MockDataLoader")
        self.assertEqual(parsed[1][1][0].__class__.__name__, "MockDataLoader")
        self.assertNotEqual(parsed[1][1][0], parsed[1][0][0])
        self.assertEqual(parsed[1][0][1].__class__.__name__, "MockPipeline")
        self.assertEqual(parsed[1][0][1].steps[0].a, 1)
        self.assertEqual(parsed[1][1][1].steps[0].a, 2)

    def test_get_additional_args(self):
        parser = SimpleDictParser()
        kwargs = parser.get_additional_kwargs(None, {})
        self.assertEqual(
            kwargs,
            {
                "extra_paths": {
                    "help": "Paths to files containing custom callables.",
                    "nargs": "+",
                    "default": [],
                    "required": False,
                }
            },
        )

    def test_set_additional_args(self):
        parser = SimpleDictParser()
        self.assertEqual(len(parser.all_available_callables), 0)
        parser.set_additional_kwargs({})
        self.assertNotEqual(len(parser.all_available_callables), 0)
        parser.set_additional_kwargs({"extra_paths": [__file__]})
        self.assertIn("SimpleDictParserTest", parser.all_available_callables)
