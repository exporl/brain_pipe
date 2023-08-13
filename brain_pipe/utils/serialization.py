"""Serialization utilities."""
import pickle


def pickle_dump_wrapper(path, obj):
    """Wrap around pickle.dump.

    Parameters
    ----------
    path: str
        The path to dump the object to.
    obj: Any
        The object to dump.
    """
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


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
    with open(path, "rb") as f:
        return pickle.load(f)
