"""Utilities to find classes in a package."""
import inspect
import os
from pydoc import importfile


class Finder:
    """Utility to find classes in a package."""

    def __init__(self):
        """Create a finder."""
        self.to_find = None

    def __call__(self, to_find=None, paths=None, attach_to_main=False):
        """Find a(ll) classes in the brain_pipe package or other paths.

        Parameters
        ----------
        to_find: Optional[Any]
            The class to find subclasses of. If None, all classes are returned.
        paths: Optional[Sequence[Optional[str]]]
            The paths to search for classes. If None, the brain_pipe package is
            searched.
        attach_to_main: bool
            Whether to attach the found classes to the main module. This is useful
            if the scripts that are present in :param:`paths` can be used to
            run the pipelines as well. Running the script from directly versus from
            a parsed file will then cause problems with pickle, because the
            classes don't share the same module, i.e. __main__. Objects/classes from
            the :mod:`brain_pipe` module are never attached to __main__, regardless
            of this parameter.


        Returns
        -------
        Dict[str, Any]
            A dictionary containing all found classes. Keys correspond to the
            class names.
        """
        self.to_find = to_find
        results = {}
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if paths is None:
            paths = [root_dir]

        # Walk through the package and inspect all modules.
        for path in paths:
            if path is None:
                path = root_dir
            if path == root_dir:
                _attach_to_main = False
            else:
                _attach_to_main = attach_to_main

            # Just load it from a path
            if os.path.isfile(path):
                results.update(self.load_from_path(path, _attach_to_main))
            # Load it from folders containing python files.
            else:
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if file.endswith(".py"):
                            results.update(
                                self.load_from_path(
                                    os.path.join(root, file), _attach_to_main
                                )
                            )
        return results

    def predicate_fn(self, member):
        """Predicate function to filter classes.

        Parameters
        ----------
        member: Any
            The member to be filtered.

        Returns
        -------
        bool
            Whether the member should be included.
        """
        return self.filter_fn(member, self.to_find)

    def filter_fn(self, member, to_find):
        """Filter function to filter classes.

        Parameters
        ----------
        member: Any
            The member to be filtered.
        to_find: Optional[Any]
            The class to find subclasses of. If None, all classes are returned.

        Returns
        -------
        bool
            Whether the member should be included.
        """
        if to_find is None:
            return not inspect.isbuiltin(member)
        elif inspect.isclass(to_find):
            return inspect.isclass(member) and issubclass(member, to_find)
        else:
            raise ValueError("'to_find' must be a class or None")

    def load_from_path(self, path, attach_to_main=False):
        """Load all classes from a path.

        Parameters
        ----------
        path: str
            The path to load the classes from.
        attach_to_main: bool
            Whether to attach the found classes to the main module. See the description
            of method :meth:`__call__` for more information.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing all found classes. Keys correspond to the
            class names.
        """
        results = {}
        module = importfile(path)
        for name, obj in inspect.getmembers(module, predicate=self.predicate_fn):
            if name.startswith("_"):
                continue
            if attach_to_main:
                import __main__

                setattr(__main__, name, obj)
            results[name] = obj
        return results

    def filter(self, previously_found, obj):
        """Filter a dictionary of previously found classes.

        Parameters
        ----------
        previously_found: Dict[str, Any]
            A dictionary containing all previously found classes. Keys correspond to the
            class names.
        obj: Any
            The object to filter the dictionary with. Only subclasses are included

        Returns
        -------
        Dict[str, Any]
            A dictionary containing all found classes. Keys correspond to the
            class names.
        """
        new_found = {}
        for key, value in previously_found.items():
            if self.filter_fn(value, obj):
                new_found[key] = value
        return new_found
