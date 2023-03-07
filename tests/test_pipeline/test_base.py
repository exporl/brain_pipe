import unittest
from collections import OrderedDict

from brain_pipe.pipeline.base import PipelineStep, Pipeline


class PipelineStepTest(unittest.TestCase):
    class MockPipelineStep(PipelineStep):
        def __call__(self, data_dict):
            return super(PipelineStepTest.MockPipelineStep, self).__call__(data_dict)

    def test_basic(self):
        step = PipelineStepTest.MockPipelineStep()
        test_d = {"a": 1, "b": "3"}
        output = step(test_d)
        self.assertEqual(test_d, output)
        test_d["a"] = 2
        self.assertEqual(output["a"], 2)

    def test_copy_data_dict(self):
        step = PipelineStepTest.MockPipelineStep(copy_data_dict=True)
        test_d = {"a": 1, "b": "3"}
        output = step(test_d)
        test_d["a"] = 2
        self.assertEqual(output["a"], 1)

    def test_parse_dict_keys(self):
        step = PipelineStepTest.MockPipelineStep()
        self.assertEqual(step.parse_dict_keys({"a": "b"}), {"a": "b"})
        self.assertEqual(step.parse_dict_keys(["a", "b"]), {"a": "a", "b": "b"})
        self.assertEqual(step.parse_dict_keys("abc"), {"abc": "abc"})
        with self.assertRaises(TypeError):
            step.parse_dict_keys({"a": "b"}, require_ordered_dict=True)
        self.assertEqual(
            step.parse_dict_keys(OrderedDict([("a", "b")]), require_ordered_dict=True),
            OrderedDict([("a", "b")]),
        )
        with self.assertRaises(TypeError):
            step.parse_dict_keys(None)
        with self.assertRaises(TypeError):
            step.parse_dict_keys(None, require_ordered_dict=True)


class PipelineTest(unittest.TestCase):
    class MockPipeline(Pipeline):
        def __call__(self, data_dict):
            data_dict = super().__call__(data_dict)
            data_dict["steps"] = self.steps
            return data_dict

    def test_steps(self):
        steps = [PipelineStepTest.MockPipelineStep()]
        pipeline = PipelineTest.MockPipeline(steps=steps)
        result = pipeline({"a": 1})
        self.assertEqual(result["steps"], steps)

    def test_copy_data_dict(self):
        steps = [PipelineStepTest.MockPipelineStep()]
        pipeline = PipelineTest.MockPipeline(steps=steps, copy_data_dict=True)
        start_dict = {"a": 1}
        result = pipeline(start_dict)
        self.assertEqual(result["steps"], steps)
        self.assertEqual(result["a"], 1)
        self.assertNotEqual(result, start_dict)

        pipeline = PipelineTest.MockPipeline(steps=steps, copy_data_dict=False)
        result = pipeline(start_dict)
        self.assertEqual(result, start_dict)
