"""Simple dataloader that loads data based on a glob pattern."""
import glob
from typing import Iterator, Any, Dict

from brain_pipe.dataloaders.base import DataLoader


class GlobLoader(DataLoader):
    """Simple dataloader that loads data based on a glob pattern."""

    def __init__(
        self,
        glob_patterns,
        filter_fns=tuple(),
        glob_fn=glob.iglob,
        key="path",
        chain=all,
        has_length=True,
    ):
        """Initialize the GlobLoader.

        Parameters
        ----------
        glob_patterns : List[str]
            The glob patterns to use for loading the data.
        filter_fns : Sequence[Callable[[str], bool]]
            A sequence of functions that return True if the file should be
            included.
        glob_fn : Callable[[str], Sequence[str]]
            The function to use for globbing.
        key : str
            The key to use for the data_dict.
        chain: Callable[[Sequence[bool]], bool]
            The function to use for chaining the filter functions.
        """
        super().__init__(has_length=has_length)
        self.glob_patterns = glob_patterns
        self.filter_fns = filter_fns
        self.glob_fn = glob_fn
        self.key = key
        self.chain = chain

    def path_to_data_dict(self, path):
        """Convert a path to a data_dict.

        Parameters
        ----------
        path: str
            The path to convert.

        Returns
        -------
        Dict[str, Any]
        """
        return {self.key: path}

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over the files.

        Returns
        -------
        Iterator[Dict[str, Any]]
            An iterator that produces data_dicts with 'key' as key and the path as
            value.
        """
        for pattern in self.glob_patterns:
            for el in self.glob_fn(pattern):
                if self.chain([filter_fn(el) for filter_fn in self.filter_fns]):
                    yield self.path_to_data_dict(el)
