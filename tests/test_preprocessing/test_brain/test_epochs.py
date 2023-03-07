import unittest

from brain_pipe.preprocessing.brain.epochs import SplitEpochs


class SplitEpochsTest(unittest.TestCase):
    def test_data_dict_copy(self):
        se = SplitEpochs()
        output = se.data_dict_copy({"a": 1, "b": 2}, ["a", "b"])
        self.assertEqual(output, {})

        output = se.data_dict_copy({"a": 1, "b": 2}, ["b"])
        self.assertEqual(output, {"a": 1})

        output = se.data_dict_copy({"a": 1, "b": 2}, [])
        self.assertEqual(output, {"a": 1, "b": 2})

    def test_multiple_keys(self):
        se = SplitEpochs(["stimuli", "stimuli2"], ["data", "data2"])
        output = se.data_dict_copy({"a": 1, "b": 2}, [])
        self.assertEqual(output, {"a": 1, "b": 2})

    def test_default(self):
        se = SplitEpochs()
        d = {
            "stimuli": [{"a": 1}, {"b": 2}],
            "data": [[1, 2, 3], [4, 5, 6]],
        }
        output = se(d)
        self.assertEqual(
            output, [{"data": [1, 2, 3], "a": 1}, {"data": [4, 5, 6], "b": 2}]
        )
