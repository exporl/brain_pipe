import os.path
import unittest

from brain_pipe.dataloaders.path import GlobLoader


class GlobLoaderTest(unittest.TestCase):
    def test_path_to_data_dict(self):
        path = "test_data/test_path/test_path.py"

        loader = GlobLoader([])
        output = loader.path_to_data_dict(path)
        self.assertEqual(output, {"path": path})

        loader = GlobLoader([], key="something")
        output = loader.path_to_data_dict(path)
        self.assertEqual(output, {"something": path})

    def test_glob_loader(self):
        test_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        glob_pattern = os.path.join(test_folder, "test_data", "*.bdf")
        loader = GlobLoader([glob_pattern])
        output = list(map(lambda x: x, loader))
        self.assertEqual(
            output,
            [{"path": os.path.join(test_folder, "test_data", "Newtest17-256.bdf")}],
        )

    def _filter_fn_setup(self, test_folder, chain):
        glob_pattern = os.path.join(test_folder, "test_data", "*")
        loader = GlobLoader(
            [glob_pattern],
            filter_fns=[
                lambda x: "5" in os.path.basename(x),
                lambda x: "New" in os.path.basename(x),
            ],
            chain=chain,
        )
        return list(map(lambda x: x, loader))

    def test_glob_loader_filter_fns_all(self):
        test_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output = self._filter_fn_setup(test_folder, all)
        self.assertEqual(
            output,
            [{"path": os.path.join(test_folder, "test_data", "Newtest17-256.bdf")}],
        )

    def test_glob_loader_filter_fns_any(self):
        test_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output = self._filter_fn_setup(test_folder, any)
        self.assertEqual(
            sorted(output, key=lambda x: x["path"]),
            [
                {
                    "path": os.path.join(
                        test_folder, "test_data", "Newtest17-2048.bdf.gz"
                    )
                },
                {"path": os.path.join(test_folder, "test_data", "Newtest17-256.bdf")},
                {
                    "path": os.path.join(
                        test_folder, "test_data", "t_audiobook_5_1.npz.gz"
                    )
                },
            ],
        )
