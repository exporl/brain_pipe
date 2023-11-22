"""Multiprocessing utilities."""
import abc
import logging

import multiprocessing


class ProgressCallbackFn(abc.ABC):
    """Callback function for multiprocessing to track the progress."""

    def __init__(self, total=None, current_count=0):
        """Create a new SimpleCallbackFn instance."""
        self.total = total
        self.current_count = current_count

    @abc.abstractmethod
    def __call__(self, result):
        """Process the result of the multiprocessing.

        Parameters
        ----------
        result: Any
            Single result of the multiprocessing.
        """


class SimpleCallbackFn(ProgressCallbackFn):
    """Simple callback function for multiprocessing.

    Is used to track the progress of the multiprocessing.
    """

    def __call__(self, result):
        """Call the SimpleCallbackFn instance.

        Parameters
        ----------
        result: Any
            Single result of the multiprocessing.
        """
        self.current_count += 1
        logging.info(f"Progress {self.current_count}/{self.total}.")


class MultiprocessingSingleton:
    """Singleton class for multiprocessing."""

    manager = multiprocessing.Manager()
    locks = {}

    to_clean = []

    @classmethod
    def get_map_fn(
        cls,
        nb_processes,
        callback: ProgressCallbackFn = SimpleCallbackFn(),
        maxtasksperchild=None,
    ):
        """Create a map function for multiprocessing.

        Parameters
        ----------
        nb_processes: int
            Number of processes to use. If -1, use all available cores.
            If 0, do not use multiprocessing.
        callback: ProgressCallbackFn
            Callback function to track the progress of the multiprocessing.
        maxtasksperchild: Optional[int]
            Maximum number of tasks per child process.

        Returns
        -------
        Callable
            Map function.
        """
        if nb_processes != 0:
            if nb_processes == -1:
                nb_processes = multiprocessing.cpu_count()
            pool = multiprocessing.Pool(nb_processes, maxtasksperchild=maxtasksperchild)
            cls.to_clean += [pool]

            def dummy_map_fn(fn, iterable):
                # Reset the current count
                callback.current_count = 0
                try:
                    callback.total = len(iterable)
                except TypeError:
                    callback.total = "?"
                async_results = [
                    pool.apply_async(fn, args=[item], callback=callback)
                    for item in iterable
                ]
                results = [async_result.get() for async_result in async_results]
                return results

        else:

            def dummy_map_fn(fn, iterable):
                try:
                    callback.total = len(iterable)
                except TypeError:
                    callback.total = "?"
                results = []
                for item in iterable:
                    results.append(fn(item))
                    callback(item)
                return results

        return dummy_map_fn

    @classmethod
    def clean(cls):
        """Clean up the multiprocessing pools."""
        for pool in cls.to_clean:
            pool.close()
            pool.join()
        cls.to_clean = []

    @classmethod
    def get_lock(cls, id_str):
        """Create or get a lock for multiprocessing.

        Parameters
        ----------
        id_str: str
            Identifier for the lock. If the lock does not already exist in self.locks,
            it will be created and added to self.locks.

        Returns
        -------
        multiprocessing.Lock
        """
        if id_str not in cls.locks:
            cls.locks[id_str] = cls.manager.Lock()
        return cls.locks[id_str]
