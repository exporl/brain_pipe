import unittest

from brain_pipe.save.base import Save


class BaseSaveTest(unittest.TestCase):
    class BaseSaveMockup(Save):
        def is_already_done(self, data_dict):
            return data_dict["is_already_done"]

        def is_reloadable(self, data_dict):
            return data_dict["is_reloadable"]

        def reload(self, data_dict):
            return data_dict["reload"]

        def __call__(self, data_dict):
            return data_dict

    def test_is_already_done(self):
        self.assertTrue(
            self.BaseSaveMockup().is_already_done({"is_already_done": True})
        )
        self.assertFalse(
            self.BaseSaveMockup().is_already_done({"is_already_done": False})
        )

    def test_is_reloadable(self):
        self.assertTrue(self.BaseSaveMockup().is_reloadable({"is_reloadable": True}))
        self.assertFalse(self.BaseSaveMockup().is_reloadable({"is_reloadable": False}))

    def test_reload(self):
        test_dict = {"a": 1, "b": 2}
        self.assertEqual(self.BaseSaveMockup().reload({"reload": test_dict}), test_dict)
