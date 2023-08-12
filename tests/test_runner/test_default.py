import logging
import types
import unittest

from brain_pipe.runner.default import DefaultRunner
from brain_pipe.utils.multiprocess import MultiprocessingSingleton


class DefaultRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        logging.shutdown()
        logging.getLogger().handlers = []

    def test_default_options(self):
        runner = DefaultRunner()
        self.assertIsInstance(runner.map_fn, types.LambdaType)
        logger = logging.getLogger()
        self.assertEqual(len(logger.handlers), 1)

    def test_nb_processes(self):
        MultiprocessingSingleton.clean()
        runner = DefaultRunner(nb_processes=2)
        self.assertEqual(len(MultiprocessingSingleton.to_clean), 1)

    def test_logging_conf(self):
        runner = DefaultRunner(logging_config={"level": logging.DEBUG})
        logger = logging.getLogger()
        self.assertEqual(len(logger.handlers), 1)
        self.assertEqual(logger.level, logging.DEBUG)

    def test_logging_conf_callable(self):
        def set_debug():
            logging.getLogger().setLevel(logging.DEBUG)

        runner = DefaultRunner(logging_config=set_debug)
        logger = logging.getLogger()
        self.assertEqual(logger.level, logging.DEBUG)

    def test_logging_conf_invalid(self):
        with self.assertRaises(ValueError):
            runner = DefaultRunner(logging_config=1)

    def tearDown(self) -> None:
        logging.shutdown()
