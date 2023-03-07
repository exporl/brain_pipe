import unittest

from brain_pipe.dataloaders.base import DataLoader


class DataLoaderTest(unittest.TestCase):
    class MockDataLoader(DataLoader):
        def __iter__(self):
            return [1, 2, 3].__iter__()

    def test_iter(self):
        """Test the __iter__ method."""
        loader = self.MockDataLoader()
        self.assertEqual(list(loader), [1, 2, 3])
        self.assertEqual(next(loader), 1)
        self.assertEqual(next(loader), 2)
        self.assertEqual(next(loader), 3)
        with self.assertRaises(StopIteration):
            next(loader)
        self.assertEqual([x for x in loader], [1, 2, 3])
        self.assertEqual(list(map(lambda x: x, loader)), [1, 2, 3])

    def test_len(self):
        """Test the __len__ method."""
        loader = self.MockDataLoader()
        self.assertEqual(len(loader), 3)
        loader.has_length = False
        with self.assertRaises(TypeError):
            len(loader)
        loader.has_length = True
        self.assertEqual(len(loader), 3)
