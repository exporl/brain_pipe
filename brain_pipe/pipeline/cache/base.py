"""Module for caching intermediate results of a PreprocessingPipeline."""
import abc
import os
from typing import Optional, Callable, Dict, Any, Sequence, Union

from brain_pipe.pipeline.base import PipelineStep
from brain_pipe.utils.serialization import (
    pickle_dump_wrapper,
    pickle_load_wrapper,
)


class PipelineCache(abc.ABC):
    """A cache to store intermediate data_dicts of a PreprocessingPipeline."""

    def __init__(
        self,
        cache_root: str,
        cache_key: str = "cache",
        previous_cache_folder_key: str = "previous_cache",
        serializer_fn: Callable[[Any, str], None] = pickle_dump_wrapper,
        deserializer_fn: Callable[[str], Any] = pickle_load_wrapper,
    ):
        """Create a PipelineCache.

        Parameters
        ----------
        cache_root: str
            Path to the cache folder
        cache_key: str
            Key to use to store the cache location
        previous_cache_folder_key: str
            Key to store the previous location of the cache
        serializer_fn: Callable[[Any, str], None]
            Function that can serialize data at a certain file location.
            The first argument is the data, second argument is the file
            location.
        deserializer_fn: Callable[[str], Any]
            Function that can deserialize data at a certain file location.
            The first argument is the file location. The return value is the
            deserialized data.
        """
        self.cache_root = cache_root
        os.makedirs(self.cache_root, exist_ok=True)
        self.cache_key = cache_key
        self.previous_cache_folder_key = previous_cache_folder_key
        self.serializer_fn = serializer_fn
        self.deserializer_fn = deserializer_fn

    @abc.abstractmethod
    def predict_filenames_from_data_dict(
        self, data_dict: Dict[str, Any]
    ) -> Sequence[Sequence[str]]:
        """Predict possible filenames from a data_dict.

        Parameters
        ----------
        data_dict: Dict[str, Any]
            Dictionary containing the data

        Returns
        -------
        Sequence[Sequence[str]]
            Sequence of Sequence of possible filenames. In the innermost
            sequence, all files should (have been) created at the same time
        """
        pass

    @abc.abstractmethod
    def predict_filenames_from_previous_filename(
        self, previous_filename: str
    ) -> Sequence[str]:
        """Predict possible filenames from a previous filename.

        Parameters
        ----------
        previous_filename: str
            Filename of the previous step

        Returns
        -------
        Sequence[str]
            Sequence of possible filenames
        """
        pass

    @abc.abstractmethod
    def get_filename(self, data_dict: Dict[str, Any]) -> str:
        """Get the filename for a data_dict.

        Parameters
        ----------
        data_dict: dict

        Returns
        -------
        str
            Name of cache file
        """
        pass

    @abc.abstractmethod
    def get_foldername(self, step: PipelineStep, step_index: Optional[int]) -> str:
        """Get the foldername for a step.

        Parameters
        ----------
        step: PipelineStep
            PreprocessingStep that is being cached
        step_index: Optional[int]
            Index of the step in the pipeline. None if the step is not part of
            a pipeline.


        Returns
        -------
        str
            Name of cache folder
        """
        pass

    def get_cache_dict(
        self, path: str, step: PipelineStep, step_index: Optional[int]
    ) -> Dict[str, Any]:
        """Create a dictionary containing the cache information.

        Parameters
        ----------
        path: str
            Path to the cache file
        step: PipelineStep
            PipelineStep that is being cached
        step_index: Optional[int]
            Index of the step in the pipeline. None if the step is not part of
            a pipeline.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the cache information
        """
        return {
            self.cache_key: os.path.basename(path),
            self.previous_cache_folder_key: self.get_foldername(step, step_index),
        }

    def load(self, path: str) -> Dict[str, Any]:
        """Load a cache file.

        Parameters
        ----------
        path: str
            Path to the cache file
        Returns
        -------
        Dict[str, Any]
            Dictionary containing the cache information
        """
        return self.deserializer_fn(path)

    def load_from_data_dict(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Load a cache file from a data_dict.

        Parameters
        ----------
        data_dict: Dict[str, Any]
            Dictionary containing the cache information

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the cache information
        """
        previous_cache_path = os.path.join(
            self.cache_root,
            data_dict[self.previous_cache_folder_key],
            self.get_filename(data_dict),
        )
        return self.load(previous_cache_path)

    def save(self, path: str, data_dict: Dict[str, Any]):
        """Save a cache file.

        Parameters
        ----------
        path: str
            Path to save the cache file to.
        data_dict: Dict[str, Any]
            Dictionary containing the cache information

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the cache information
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.serializer_fn(data_dict, path)

    def get_existing_cache_paths(
        self,
        step: PipelineStep,
        data_dict: Dict[str, Any],
        step_index: Optional[int],
    ) -> Sequence[str]:
        """Get all existing cache paths for a step.

        Parameters
        ----------
        step: PipelineStep
            PipelineStep that is being cached
        data_dict: Dict[str, Any]
            Dictionary containing the data.
        step_index: Optional[int]
            Index of the step in the pipeline. None if the step is not part of
            a pipeline.

        Returns
        -------
        Sequence[str]
            Sequence of possibly existing cache paths
        """
        if self.cache_key in data_dict:
            previous_filename = data_dict[self.cache_key]
            return self.find_existing_cache_from_previous_filename(
                step, previous_filename, step_index
            )
        else:
            return self.find_existing_cache_from_data_dict(step, data_dict, step_index)

    def get_path(
        self,
        step: PipelineStep,
        data_dict: Dict[str, Any],
        step_index: Optional[int],
    ) -> str:
        """Create a path for a cache file.

        Parameters
        ----------
        step: PipelineStep
            PipelineStep that is being cached
        data_dict: Dict[str, Any]
            Dictionary containing the data.
        step_index: Optional[int]
            Index of the step in the pipeline. None if the step is not part of
            a pipeline.

        Returns
        -------
        str
            Path to the cache file
        """
        return os.path.join(
            self.cache_root,
            self.get_foldername(step, step_index),
            self.get_filename(data_dict),
        )

    def _predict_paths(
        self,
        step: PipelineStep,
        step_index: Optional[int],
        predict_fn: Callable[[Union[Dict[str, Any], str]], Sequence[Sequence[str]]],
        data_dict_or_filename: Union[Dict[str, Any], str],
    ) -> Sequence[str]:
        predicted_paths = []
        for filename_group in predict_fn(data_dict_or_filename):
            temp_paths = []
            for filename in filename_group:
                temp_paths += [
                    os.path.join(
                        self.cache_root, self.get_foldername(step, step_index), filename
                    )
                ]
            predicted_paths += [temp_paths]
        return predicted_paths

    def _find_existing_cache(
        self,
        step: PipelineStep,
        step_index: Optional[int],
        predict_fn: Callable[[Union[Dict[str, Any], str]], Sequence[Sequence[str]]],
        data_dict_or_filename: Union[Dict[str, Any], str],
    ) -> Sequence[str]:
        for path_group in self._predict_paths(
            step, step_index, predict_fn, data_dict_or_filename
        ):
            path_group_exists = True
            for path in path_group:
                path_group_exists = path_group_exists and os.path.exists(path)
            if path_group_exists:
                return path_group
        return []

    def predict_paths_from_data_dict(
        self,
        step: PipelineStep,
        data_dict: Dict[str, Any],
        step_index: Optional[int],
    ) -> Sequence[str]:
        """Predict the paths of the cache files that will be created by a step.

        Parameters
        ----------
        step: PipelineStep
            PipelineStep that is being cached
        data_dict:
            Dictionary containing the data. The path will be predicted from this.
        step_index: Optional[int]
            Index of the step in the pipeline. None if the step is not part of
            a pipeline.

        Returns
        -------
        Sequence[str]
            Sequence of predicted cache paths
        """
        return self._predict_paths(
            step, step_index, self.predict_filenames_from_data_dict, data_dict
        )

    def predict_paths_from_previous_filename(
        self, step: PipelineStep, previous_filename: str, step_index: Optional[int]
    ) -> Sequence[str]:
        """Predict the paths of the cache files that will be created by a step.

        Parameters
        ----------
        step: PipelineStep
            PipelineStep that is being cached
        previous_filename: str
            Path to the previous cache file. The path will be predicted from this.
        step_index: Optional[int]
            Index of the step in the pipeline. None if the step is not part of
            a pipeline.

        Returns
        -------
        Sequence[str]
            Sequence of predicted cache paths
        """
        return self._predict_paths(
            step,
            step_index,
            self.predict_filenames_from_previous_filename,
            previous_filename,
        )

    def find_existing_cache_from_data_dict(
        self,
        step: PipelineStep,
        data_dict: Dict[str, Any],
        step_index: Optional[int],
    ) -> Sequence[str]:
        """Find existing cache files for a step.

        Parameters
        ----------
        step: PipelineStep
            PipelineStep that is being cached
        data_dict: Dict[str, Any]
            Dictionary containing the data.
        step_index: Optional[int]
            Index of the step in the pipeline. None if the step is not part of
            a pipeline.

        Returns
        -------
        Sequence[str]
            Sequence of possibly existing cache paths
        """
        return self._find_existing_cache(
            step, step_index, self.predict_filenames_from_data_dict, data_dict
        )

    def find_existing_cache_from_previous_filename(
        self, step: PipelineStep, previous_filename: str, step_index: Optional[int]
    ) -> Sequence[str]:
        """Find existing cache files for a step.

        Parameters
        ----------
        step: PipelineStep
            PipelineStep that is being cached
        previous_filename: str
            Path to the previous cache file.
        step_index: Optional[int]
            Index of the step in the pipeline. None if the step is not part of
            a pipeline.

        Returns
        -------
        Sequence[str]
            Sequence of possibly existing cache paths
        """
        return self._find_existing_cache(
            step,
            step_index,
            self.predict_filenames_from_previous_filename,
            previous_filename,
        )
