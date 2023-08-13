"""Serialization utilities."""
import pickle
import sys


def pickle_dump_wrapper(path, obj):
    """Wrap around pickle.dump.

    Parameters
    ----------
    path: str
        The path to dump the object to.
    obj: Any
        The object to dump.
    """
    pickle_mod = pickle
    if sys.version_info < (3, 8):
        import pickle5

        pickle_mod = pickle5
    with open(path, "wb") as f:
        pickle_mod.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load_wrapper(path):
    """Wrap around pickle.load.

    Parameters
    ----------
    path: str
        The path to load the object from.

    Returns
    -------
    Any
        The loaded object.
    """
    pickle_mod = pickle
    if sys.version_info < (3, 8):
        import pickle5

        pickle_mod = pickle5
    with open(path, "rb") as f:
        return pickle_mod.load(f)
