"""Base class for preprocessing steps."""
import abc
import copy
import typing

from collections import OrderedDict
from typing import Sequence, Any, Dict, Union, Mapping


class PipelineStep(abc.ABC):
    """A preprocessing step.

    This class is the base PreprocessingStep class.

    The PreprocessingStep is the base building block of all preprocessing pipelines in
    this repository.

    In essence, it is a functor (a class that can be called like a function) that takes
    a data dictionary and performs some preprocessing on it. The data dictionary can be
    modified in-place or a copy can be returned.

    Some convenient methods are provided to make it easier to write preprocessing
    steps like the :meth:`parse_dict_keys` method that can be used to parse a key or a
    sequence of keys.

    If you want to create your own PreprocessingStep, you can inherit from this class
    (or another, existing class). For more info about the structure of
    PreprocessingSteps and how to write them, see :ref:`pipeline`.

    .. image:: ../../_images/simple_pipeline.svg
       :width: 100%
    """

    def __init__(self, copy_data_dict=False):
        """Initialize the PreprocessingStep.

        Parameters
        ----------
        copy_data_dict: bool
            Controls whether the input data dict is copied before any
            operations are applied. If False, the data_dict will be modified
            in place.
        """
        self.copy_data_dict = copy_data_dict

    @abc.abstractmethod
    def __call__(
        self, data_dict: Dict[str, Any]
    ) -> Union[Dict[str, Any], Sequence[Dict[str, Any]]]:
        """Apply a preprocessing step to a data dictionary.

        Parameters
        ----------
        data_dict: Dict[str, Any]
            A dictionary containing the data to be preprocessed.
            This dictionary is modified in-place and returned.


        Returns
        -------
        Union[Dict[str, Any], Sequence[Dict[str, Any]]]
            Preprocessed data in a dictionary format. Can be a single
            dictionary or a sequence of dictionaries.


        Notes
        -----
        In general, it is a good idea to keep the structure of data_dict as
        flat as possible, as this makes it easier to write interoperable
        preprocessing steps.
        """
        if self.copy_data_dict:
            return copy.deepcopy(data_dict)
        else:
            return data_dict

    def parse_dict_keys(
        self,
        key: Union[str, Sequence[str], Mapping[str, str]],
        name="key",
        require_ordered_dict=False,
    ) -> typing.OrderedDict[str, str]:
        """Parse a key or a sequence of keys.

        Parameters
        ----------
        key: Union[str, Sequence[str], Mapping[str,str]]
            A key or a sequence of keys.
        name: str
            The name of the key. Used for error messages.
        require_ordered_dict: bool
            If True, the key must be an OrderedDict. If False, the key can
            also be an ordinary dict.

        Returns
        -------
        OrderedDict[str, str]
            A mapping of input keys to output keys.

        Raises
        ------
        TypeError
            If the key is not a string, a sequence of strings or a mapping of
            strings.
            If the key is a mapping but require_ordered_dict is True and the
            mapping is not an OrderedDict.
        """
        if isinstance(key, str):
            return OrderedDict([(key, key)])
        elif isinstance(key, Mapping):
            if require_ordered_dict and not isinstance(key, OrderedDict):
                raise TypeError(
                    f"If the '{name}' is a mapping, it must be an ordered mapping "
                    "(e.g. not an ordinary `dict` but an `OrderedDict`)."
                )
            return OrderedDict(key.items())
        elif isinstance(key, Sequence):
            return OrderedDict([(k, k) for k in key])
        else:
            extra_msg = ""
            if require_ordered_dict:
                extra_msg = "n ordered"
            raise TypeError(
                f"The '{name}' must be a string, a sequence of strings or "
                f"a{extra_msg} mapping of strings."
            )


class Pipeline(PipelineStep, abc.ABC):
    """The base pipeline class.

    A pipeline represents a sequence of preprocessing steps (``steps``) that are
    applied in order to a data dictionary. The output of one step is the input of the
    next step.

    See Also
    --------
    :ref:`pipeline` - For more information about pipelines.
    :class:`PipelineStep` - For more information about preprocessing steps.
    :class:`brain_pipe.pipeline.default.DefaultPipleine` - A pipeline that can be used
     as a template.

    """

    def __init__(self, steps: Sequence[PipelineStep], *args, **kwargs):
        """Initialize the pipeline.

        Parameters
        ----------
        steps: Sequence[PipelineStep]
            A sequence of PipelineSteps.
        args:
            Additional arguments for the PipelineStep class.
        kwargs:
            Additional keyword arguments for the PipelineStep class.
        """
        super().__init__(*args, **kwargs)
        self.steps = steps
