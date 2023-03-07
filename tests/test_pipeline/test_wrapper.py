import unittest

from brain_pipe.pipeline.wrapper import Wrapper


class WrapperTest(unittest.TestCase):
    def test_key_none(self):
        """Test the key=None case."""

        def test_fn(d):
            d["h"] = 9
            return d

        wrapper = Wrapper(test_fn)
        output = wrapper({})
        self.assertEqual(output["h"], 9)

    def test_keys_not_none(self):
        """Test keys is not None case."""
        wrapper = Wrapper(lambda a: a + 3, key={"a": "b"})
        output = wrapper({"a": 3})
        self.assertEqual(output["a"], 3)
        self.assertEqual(output["b"], 6)
