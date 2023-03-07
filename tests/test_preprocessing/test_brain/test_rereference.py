import unittest

from brain_pipe.preprocessing.brain.rereference import CommonAverageRereference


class CommonAverageRereferenceTest(unittest.TestCase):
    def test_rereference(self):
        ref = CommonAverageRereference()
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        output = ref({"data": data})
        self.assertTrue(
            (
                output["data"] == [[-3.0, -3.0, -3.0], [0.0, 0.0, 0.0], [3.0, 3.0, 3.0]]
            ).all()
        )

        ref = CommonAverageRereference(axis=1)
        output = ref({"data": data})
        self.assertTrue(
            (
                output["data"] == [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]]
            ).all()
        )
