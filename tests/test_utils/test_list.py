import unittest

from brain_pipe.utils.list import flatten, wrap_in_list


class ListUtilsTest(unittest.TestCase):
    def test_flatten(self):
        flat_list = flatten([1, [2, [3, 4, [5], 6]]])
        self.assertEqual(flat_list, [1, 2, 3, 4, 5, 6])

    def test_single_obj_to_list(self):
        self.assertEqual(wrap_in_list(1), [1])
        self.assertEqual(wrap_in_list([1]), [1])
        self.assertEqual(wrap_in_list((1, 2)), (1, 2))
        self.assertEqual(wrap_in_list([]), [])
        self.assertEqual(wrap_in_list(None), [None])
