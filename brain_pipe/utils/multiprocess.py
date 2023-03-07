"""Multiprocessing utilities."""
import logging

import multiprocessing


class SimpleCallbackFn:
    """Simple callback function for multiprocessing.

    Is used to track the progress of the multiprocessing.
    """

    def __init__(self):
        """Create a new SimpleCallbackFn instance."""
        self.total = None
        self.current_count = 0

    def __call__(self, x):
        """Call the SimpleCallbackFn instance.

        Parameters
        ----------
        x: Any
            Dummy parameter.
        """
        self.current_count += 1
        logging.info(f"Progress {self.current_count}/{self.total}.")


class MultiprocessingSingleton:
    """Singleton class for multiprocessing."""

    manager = multiprocessing.Manager()

    to_clean = []

    @classmethod
    def get_map_fn(
        cls, nb_processes, callback=SimpleCallbackFn(), maxtasksperchild=None
    ):
        """Create a map function for multiprocessing.

        Parameters
        ----------
        nb_processes: int
            Number of processes to use. If -1, use all available cores.
            If 0, do not use multiprocessing.
        callback: Callable
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
