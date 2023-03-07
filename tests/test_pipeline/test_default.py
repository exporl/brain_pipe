import unittest
from typing import Dict, Any

from brain_pipe.pipeline.base import PipelineStep
from brain_pipe.pipeline.default import DefaultPipeline
from brain_pipe.save.base import Save


class PreprocessingPipelineTest(unittest.TestCase):
    class MockPipelineStep(PipelineStep):
        def __init__(self, d):
            super().__init__()
            self.d = d

        def __call__(self, data_dict):
            for key, value in self.d.items():
                data_dict[key] = value
            return data_dict

    def setUp(self) -> None:
        self.pipeline_step = self.MockPipelineStep({"a": 3})
        self.pipeline = DefaultPipeline(
            [
                self.pipeline_step,
                self.MockPipelineStep({"b": 2}),
                self.MockPipelineStep({"a": 1}),
            ],
            previous_steps_key="previous",
        )

    def test_run_step(self):
        output = self.pipeline.run_step(self.pipeline_step, {"k": 3})
        self.assertEqual(
            output,
            {
                "k": 3,
                "a": 3,
                "previous": [
                    {
                        "copy_data_dict": False,
                        "d": {"a": 3},
                        "step_index": None,
                        "step_name": "MockPipelineStep",
                    }
                ],
            },
        )

        output = self.pipeline.run_step(self.pipeline_step, output, step_index=2)
        self.assertEqual(
            output,
            {
                "k": 3,
                "a": 3,
                "previous": [
                    {
                        "copy_data_dict": False,
                        "d": {"a": 3},
                        "step_index": None,
                        "step_name": "MockPipelineStep",
                    },
                    {
                        "copy_data_dict": False,
                        "d": {"a": 3},
                        "step_index": 2,
                        "step_name": "MockPipelineStep",
                    },
                ],
            },
        )

    def test_iterate_over_steps(self):
        output = self.pipeline.iterate_over_steps(
            self.pipeline.steps,
            {"k": 3},
        )
        self.assertEqual(
            output,
            [
                {
                    "k": 3,
                    "a": 1,
                    "previous": [
                        {
                            "copy_data_dict": False,
                            "d": {"a": 3},
                            "step_index": 0,
                            "step_name": "MockPipelineStep",
                        },
                        {
                            "copy_data_dict": False,
                            "d": {"b": 2},
                            "step_index": 1,
                            "step_name": "MockPipelineStep",
                        },
                        {
                            "copy_data_dict": False,
                            "d": {"a": 1},
                            "step_index": 2,
                            "step_name": "MockPipelineStep",
                        },
                    ],
                    "b": 2,
                }
            ],
        )

    def test_default(self):
        output = self.pipeline({"a": -2})
        self.assertEqual(
            output,
            [
                {
                    "a": 1,
                    "previous": [
                        {
                            "copy_data_dict": False,
                            "d": {"a": 3},
                            "step_index": 0,
                            "step_name": "MockPipelineStep",
                        },
                        {
                            "copy_data_dict": False,
                            "d": {"b": 2},
                            "step_index": 1,
                            "step_name": "MockPipelineStep",
                        },
                        {
                            "copy_data_dict": False,
                            "d": {"a": 1},
                            "step_index": 2,
                            "step_name": "MockPipelineStep",
                        },
                    ],
                    "b": 2,
                }
            ],
        )

    def test_continue_on_error(self):
        class MockErrorPipelineStep(PipelineStep):
            def __call__(self, data_dict):
                raise ValueError()

        self.pipeline.steps.insert(0, MockErrorPipelineStep())
        output = self.pipeline({"a": -2})
        self.assertEqual(output, {"a": -2})
        self.pipeline.on_error = self.pipeline.RAISE
        with self.assertRaises(ValueError):
            self.pipeline({"a": -2})

        self.pipeline.on_error = self.pipeline.CONTINUE
        output = self.pipeline({"a": -2})
        self.assertEqual(
            output,
            [
                {
                    "a": 1,
                    "previous": [
                        {
                            "copy_data_dict": False,
                            "step_index": 0,
                            "step_name": "MockErrorPipelineStep",
                        },
                        {
                            "copy_data_dict": False,
                            "d": {"a": 3},
                            "step_index": 1,
                            "step_name": "MockPipelineStep",
                        },
                        {
                            "copy_data_dict": False,
                            "d": {"b": 2},
                            "step_index": 2,
                            "step_name": "MockPipelineStep",
                        },
                        {
                            "copy_data_dict": False,
                            "d": {"a": 1},
                            "step_index": 3,
                            "step_name": "MockPipelineStep",
                        },
                    ],
                    "b": 2,
                }
            ],
        )

        with self.assertRaises(ValueError):
            self.pipeline.on_error = "invalid"

    def test_save_step(self):
        class MockSave(Save):
            def __init__(self, done, reloadable):
                super().__init__()
                self.done = done
                self.reloadable = reloadable

            def is_already_done(self, data_dict: Dict[str, Any]) -> bool:
                return self.done

            def is_reloadable(self, data_dict: Dict[str, Any]) -> bool:
                return self.reloadable

            def reload(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
                return {"reloaded": True}

            def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
                return data_dict

        step = self.MockPipelineStep({"a": 3})
        pipeline = DefaultPipeline(
            [step, MockSave(True, True)],
        )
        self.assertEqual(pipeline({"b": 1}), [{"reloaded": True}])

        pipeline = DefaultPipeline(
            [step, MockSave(False, True)],
        )
        self.assertEqual(
            pipeline({"b": 1}),
            [
                {
                    "b": 1,
                    "a": 3,
                    "previous_steps": [
                        {
                            "copy_data_dict": False,
                            "d": {"a": 3},
                            "step_index": 0,
                            "step_name": "MockPipelineStep",
                        },
                        {
                            "copy_data_dict": False,
                            "clear_output": False,
                            "done": False,
                            "reloadable": True,
                            "step_index": 1,
                            "step_name": "MockSave",
                        },
                    ],
                }
            ],
        )

        pipeline = DefaultPipeline(
            [step, MockSave(True, False)],
        )
        self.assertEqual(pipeline({"b": 1}), [])

        pipeline = DefaultPipeline(
            [step, MockSave(False, False)],
        )
        self.assertEqual(
            pipeline({"b": 1}),
            [
                {
                    "b": 1,
                    "a": 3,
                    "previous_steps": [
                        {
                            "copy_data_dict": False,
                            "d": {"a": 3},
                            "step_index": 0,
                            "step_name": "MockPipelineStep",
                        },
                        {
                            "copy_data_dict": False,
                            "clear_output": False,
                            "done": False,
                            "reloadable": False,
                            "step_index": 1,
                            "step_name": "MockSave",
                        },
                    ],
                }
            ],
        )
