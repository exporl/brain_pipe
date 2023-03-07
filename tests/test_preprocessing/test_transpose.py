import unittest

import numpy as np

from brain_pipe.preprocessing.transpose import Transpose


class TransposeTest(unittest.TestCase):
    def test_call(self):
        transpose = Transpose("a")
        output = transpose({"a": np.array([[1, 2], [3, 4]])})
        self.assertEqual(output["a"].tolist(), np.array([[1, 3], [2, 4]]).tolist())
