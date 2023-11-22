"""List utilities."""
from typing import Union, Any, List, Tuple


def flatten(lst: Union[List[Any], Tuple[Any]]) -> List[Any]:
    """Flatten a list or tuple (recursively).

    Parameters
    ----------
    lst: Union[List[Any], Tuple[Any]]
        A list to be flattened. Sublists and tuples are also flattened.

    Returns
    -------
    List[Any]
        A flattened list.
    """
    result = []
    if isinstance(lst, (list, tuple)):
        for x in lst:
            result.extend(flatten(x))
    else:
        result.append(lst)
    return result


def wrap_in_list(obj: Any) -> List[Any]:
    """Wrap an object in a list if it is not already a list.

    Parameters
    ----------
    obj: Any
        An object to be wrapped in a list. If it is already a list, it is returned
        as is.

    Returns
    -------
    List[Any]
        A list containing the object.
    """
    if not isinstance(obj, (list, tuple)):
        obj = [obj]
    return obj
