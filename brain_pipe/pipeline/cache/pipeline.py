"""Default caching preprocessing pipeline implementation."""
import logging
from typing import Sequence, Dict, Any, Union

from brain_pipe.pipeline.base import PipelineStep
from brain_pipe.pipeline.cache.base import PipelineCache
from brain_pipe.pipeline.default import DefaultPipeline


class CachingPreprocessingPipeline(DefaultPipeline):
    """Default caching preprocessing pipeline implementation."""

    def __init__(
        self,
        steps: Sequence[PipelineStep],
        pipeline_cache: PipelineCache,
        overwrite=False,
    ):
        """Create a new CachingPreprocessingPipeline instance.

        Parameters
        ----------
        steps: Sequence[PipelineStep]
            The preprocessing steps to be run.
        pipeline_cache: PipelineCache
            The pipeline cache class to be used.
        overwrite: bool
            Whether to overwrite existing cache files.
        """
        super().__init__(steps)
        self.pipeline_cache = pipeline_cache
        self.overwrite = overwrite

    def run_step(
        self,
        step: PipelineStep,
        data_dict: Dict[str, Any],
        step_index=None,
    ) -> Union[Dict[str, Any], Sequence[Dict[str, Any]]]:
        """Run a single preprocessing step.

        Parameters
        ----------
        step: PipelineStep
            The preprocessing step to be run.
        data_dict: Dict[str, Any]
            The data dictionary to be preprocessed.
        step_index: Optional[int]
            The index of the step in the pipeline. None if the step is not part of
            a pipeline.

        Returns
        -------
        Sequence[Dict[str, Any]]
            A list containing the data dictionaries for the next step.
        """
        # Check if the passed data_dict is already cached
        existing_cache = self.pipeline_cache.get_existing_cache_paths(
            step, data_dict, step_index
        )

        # Check if there is already a cached version of the next step
        if len(existing_cache) and not self.overwrite:
            logging.info("Step was already run, skipping to next step")
            new_data_dicts = []
            # Get the cached data_dicts for the next step
            for existing_path in existing_cache:
                new_data_dicts += [
                    self.pipeline_cache.get_cache_dict(existing_path, step, step_index)
                ]
            return new_data_dicts
        elif self.overwrite:
            logging.info("Overwrite is set to True, running step again")

        # Load the data_dict from the previous step
        if self.pipeline_cache.previous_cache_folder_key in data_dict:
            previous_data_dict = self.pipeline_cache.load_from_data_dict(data_dict)
        else:
            previous_data_dict = data_dict

        new_data_dict = super(CachingPreprocessingPipeline, self).run_step(
            step, previous_data_dict
        )

        # Handle the case where the step returns a list of data_dicts
        if isinstance(new_data_dict, dict):
            new_data_dicts = [new_data_dict]
        else:
            new_data_dicts = new_data_dict

        # Save the data_dicts to the cache
        resulting_dicts = []
        for new_data_dict in new_data_dicts:
            save_path = self.pipeline_cache.get_path(step, new_data_dict, step_index)
            self.pipeline_cache.save(save_path, new_data_dict)
            logging.debug(f"Saved data_dict to {save_path}")
            resulting_dicts += [
                self.pipeline_cache.get_cache_dict(save_path, step, step_index)
            ]
        return resulting_dicts
