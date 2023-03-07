"""Code to load stimuli from a path."""
import logging
from typing import Any, Dict

from brain_pipe.pipeline.base import PipelineStep


class LoadStimuli(PipelineStep):
    """Load stimuli from paths."""

    DEFAULT_LOAD_FNS = {}

    def __init__(
        self,
        load_from={"stimulus_path": "stimulus", "trigger_path": "trigger"},
        load_fn=None,
        separator="_",
        **kwargs,
    ):
        """Create a LoadStimuli step.

        Parameters
        ----------
        load_from: Dict[str, str]
            Dictionary mapping the key in the data dictionary to a key prefix
            where the loaded data will be stored.
        load_fn: Union[Callable[[str], Dict[str, Any]], Dict[str, Callable[[str], Dict[str, Any]]]]  # noqa: E501
            Function that can load the stimulus. If a callable, it should take
            a path as input and return a dictionary with the loaded data as key-value
            pairs.
            If a dictionary is passed, the keys should be the extensions of the files
            and the values should be the load functions (see the previous point).
            If None, the default load functions (see DEFAULT_LOAD_FNS) will be used.
        separator: str
            Separator to use between the key prefix (see load_from) and the key of the
            loaded data.
        kwargs: Dict[str, Any]
            Additional keyword arguments to pass to the PreprocessingStep constructor.
        """
        super(LoadStimuli, self).__init__(**kwargs)
        self.load_from = load_from
        if load_fn is None:
            self.load_fn = self.DEFAULT_LOAD_FNS
        else:
            self.load_fn = load_fn
        self.separator = separator

    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Load the stimuli from the paths in the data dictionary.

        Parameters
        ----------
        data_dict: Dict[str, Any]
            Dictionary containing the paths to the stimuli.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the loaded stimuli.

        Raises
        ------
        ValueError
            If a key in load_from is not in the data dictionary.
        ValueError
            If a key in load_from is None.
        """
        for key, new_key in self.load_from.items():
            if key not in data_dict:
                raise ValueError(
                    f"Can't find {key} in the data dictionary. Available "
                    f"dictionary keys are {str(list(data_dict.keys()))}."
                )
            if data_dict[key] is None:
                logging.warning(f"'{key}' was None, skipping loading {data_dict}.")
                continue

            if isinstance(self.load_fn, dict):
                extension = "." + ".".join(data_dict[key].split(".")[1:])
                if extension not in self.load_fn:
                    raise ValueError(
                        f"Can't find a load function for extension {extension}. "
                        f"Available extensions are {str(list(self.load_fn.keys()))}."
                    )
                load_fn = self.load_fn[extension]
            else:
                load_fn = self.load_fn

            loaded_data = load_fn(data_dict[key])
            for loaded_key, loaded_value in loaded_data.items():
                data_dict[new_key + self.separator + loaded_key] = loaded_value
        return data_dict
