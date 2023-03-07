import os.path
import unittest

from brain_pipe.pipeline.base import PipelineStep
from brain_pipe.utils.find import Finder


class DummyPipelineStep(PipelineStep):
    pass


class _IgnoreMe:
    pass


class FinderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.finder = Finder()

    def test_all(self):
        self.finder = Finder()

        # At the time of this comment, there are 90 things that could be found
        # in the brain_pipe package. This number will change as the package
        # grows. Therefore, only weak assertions are made here.
        self.assertTrue(len(self.finder()) >= 90)

        self.assertTrue(len(self.finder(None, [None])) >= 90)

    def test_pipeline_step(self):
        steps = self.finder(PipelineStep, [os.path.abspath(__file__)])
        self.assertEqual(len(steps), 2)
        self.assertTrue("DummyPipelineStep" in steps)
        self.assertTrue("PipelineStep" in steps)

    def test_wrong_to_find(self):
        with self.assertRaises(ValueError):
            self.finder("abc", [os.path.abspath(__file__)])

    def test_ignore_underscored(self):
        steps = self.finder(_IgnoreMe, [os.path.abspath(__file__)])
        self.assertEqual(len(steps), 0)
