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

    class MockSave(Save):
        def __init__(self, done, reloadable, clear_output=False):
            super().__init__(clear_output=clear_output)
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

    def test_save_step_done_and_reloadable(self):
        # Reload a reloadable step
        step = self.MockPipelineStep({"a": 3})
        pipeline = DefaultPipeline(
            [step, self.MockSave(True, True)],
        )
        self.assertEqual(pipeline({"b": 1}), [{"reloaded": True}])

    def test_save_step_not_done_and_reloadable(self):
        step = self.MockPipelineStep({"a": 3})
        # A step that is reloadable but not done
        # Shouldn't happen in practice, but still
        pipeline = DefaultPipeline(
            [step, self.MockSave(False, True)],
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
                            "overwrite": False,
                            "reloadable": True,
                            "step_index": 1,
                            "step_name": "MockSave",
                        },
                    ],
                }
            ],
        )

    def test_save_step_done_and_not_reloadable(self):
        step = self.MockPipelineStep({"a": 3})
        # A step that is done but not reloadable
        pipeline = DefaultPipeline(
            [step, self.MockSave(True, False)],
        )
        # No reload: True in the output
        self.assertEqual(pipeline({"b": 1}), [{"b": 1}])

    def test_save_step_not_done_nor_reloadable(self):
        step = self.MockPipelineStep({"a": 3})
        # A step that is neither done nor reloadable
        pipeline = DefaultPipeline(
            [step, self.MockSave(False, False)],
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
                            "overwrite": False,
                            "reloadable": False,
                            "step_index": 1,
                            "step_name": "MockSave",
                        },
                    ],
                }
            ],
        )

    def test_check_reload_multiple_saves_one_valid(self):
        step = self.MockPipelineStep({"a": 3})

        # Check multiple save steps, reload from 2nd
        pipeline = DefaultPipeline([])
        steps, dicts = pipeline.check_reload(
            [
                step,
                self.MockSave(True, True),
                self.MockSave(True, False),
                self.MockSave(False, False),
            ],
            {"b": 1},
        )
        self.assertEqual(len(steps), 2)
        self.assertTrue(steps[0].done)
        self.assertFalse(steps[1].reloadable)
        self.assertEqual(dicts, [{"reloaded": True}])

    def test_check_reload_multiple_saves_multiple_valid(self):
        # Check multiple save steps, reload from 2nd
        pipeline = DefaultPipeline([])
        steps, dicts = pipeline.check_reload(
            [
                self.MockSave(True, True),
                self.MockSave(True, False),
                self.MockSave(True, True),
            ],
            [{"b": 1}],
        )
        self.assertEqual(len(steps), 0)
        self.assertEqual(dicts, [{"reloaded": True}])

    def test_check_reload_multiple_saves_none_valid(self):
        # Check multiple save steps, reload from 2nd
        pipeline = DefaultPipeline([])
        steps, dicts = pipeline.check_reload(
            [
                self.MockSave(False, True),
                self.MockSave(True, False),
                self.MockSave(False, False),
            ],
            [{"b": 1}],
        )
        self.assertEqual(len(steps), 3)
        self.assertNotIn("reloaded", [{"b": 1}])

    def test_check_reload_done_not_reloadable(self):
        pipeline = DefaultPipeline([])
        steps, dicts = pipeline.check_reload(
            [
                self.MockSave(False, True),
                self.MockSave(True, False),
            ],
            [{"b": 1}],
        )
        self.assertEqual(len(steps), 0)
        self.assertEqual(dicts, [{"b": 1}])

    def test_check_reload_done_not_reloadable_clear_output(self):
        pipeline = DefaultPipeline([])
        steps, dicts = pipeline.check_reload(
            [
                self.MockSave(False, True),
                self.MockSave(True, False, clear_output=True),
            ],
            [{"b": 1}],
        )
        self.assertEqual(len(steps), 0)
        self.assertEqual(dicts, [{}])
