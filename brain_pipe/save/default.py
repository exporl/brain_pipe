"""Default save class."""
import abc
import copy
import gc
import json
import logging
import os
from typing import Any, Dict, Optional, Union, Callable, Mapping

import numpy as np

from brain_pipe.pipeline.cache.base import (
    pickle_dump_wrapper,
    pickle_load_wrapper,
)
from brain_pipe.save.base import Save
from brain_pipe.utils.list import wrap_in_list
from brain_pipe.utils.multiprocess import MultiprocessingSingleton

# Shorthand interfaces.
CheckInterface = Callable[[Dict[str, Any], str, Dict[str, Any]], Union[str, bool]]
FilenameFnInterface = Callable[[Dict[str, Any], Optional[str], Optional[str]], str]
SaveFnInterface = Union[
    Callable[[Any, str], None], Mapping[str, Callable[[Any, str], None]], None
]
ReloadFnInterface = Union[
    Callable[[str], Any], Mapping[str, Callable[[str], Any]], None
]


def default_metadata_key_fn(data_dict: Dict[str, Any]) -> str:
    """Generate a key for the metadata.

    Parameters
    ----------
    data_dict: Dict[str, Any]
        The data dict containing the data to save.

    Returns
    -------
    str
        The key for the metadata.
    """
    if "data_path" in data_dict:
        return os.path.basename(data_dict["data_path"])

    if "stimulus_path" in data_dict and data_dict["stimulus_path"] is not None:
        return os.path.basename(data_dict["stimulus_path"])

    if "trigger_path" in data_dict and data_dict["trigger_path"] is not None:
        return os.path.basename(data_dict["trigger_path"])

    raise ValueError("No data_path or stimulus_path in data_dict.")


class DefaultFilenameFn(FilenameFnInterface):
    """Default filename function to create paths to save data."""

    SPLIT_CHAR = "/"

    def __init__(
        self,
        path_keys=("data_path", "stimulus_path"),
        other_keys=("event_info/snr",),
        separator="_-_",
        data_dict_extension=".data_dict",
        feature_extension=".npy",
    ):
        """Create a new DefaultFilenameFn instance.

        Parameters
        ----------
        path_keys: Sequence[str]
            The keys of the paths to include in the filename.
        other_keys: Sequence[str]
            The keys of other data to include in the filename.
        separator: str
            The separator to use between parts of the filename.
        data_dict_extension: str
            The extension to use when saving the entire data_dict.
        feature_extension: str
            The extension to use when saving a single feature.
        """
        self.path_keys = path_keys
        self.other_keys = other_keys
        self.separator = separator
        self.data_dict_extension = data_dict_extension
        self.feature_extension = feature_extension

    def __call__(self, data_dict, feature_name=None, set_name=None):
        """Generate a filename for the data_dict.

        Parameters
        ----------
        data_dict: Dict[str, Any]
            The data dict containing the data to save.
        feature_name: Optional[str]
            The name of the feature.
        set_name: Optional[str]
            The name of the set. If no set name is given, the set name is not
            included in the filename.

        Returns
        -------
        str
            The filename.
        """
        parts = []

        # Add indicated path keys to filename
        for path_key in self.path_keys:
            if path_key in data_dict:
                parts.append(
                    "".join(os.path.basename(data_dict[path_key]).rsplit(".")[0])
                )
        # If no feature name or set name is given, return the data_dict filename.
        if feature_name is None and set_name is None:
            return self.separator.join(parts) + self.data_dict_extension

        # Add indicated other keys to data
        for full_key in self.other_keys:
            item = data_dict
            append = True
            for key in full_key.split(self.SPLIT_CHAR):
                if key not in item:
                    append = False
                    break
                item = item[key]
            if append:
                # Has to be added as a string to be able to be used in filename
                parts.append(str(item))

        # Add the feature name as the last part of the filename, if given
        if feature_name is not None:
            parts.append(feature_name)
        # Add the set name as the first part of the filename, if given
        if set_name is not None:
            parts.insert(0, set_name)
        return self.separator.join(parts) + self.feature_extension


class AttachSave(abc.ABC):
    """Mixin class to attach a Save object."""

    def __init__(self, saver=None):
        """Initialize the AttachSaver.

        Parameters
        ----------
        saver: Optional[Save]
            The saver to use. Can be attached later with :meth:`attach_saver`, but
            that will create a new object.

        """
        self.saver = saver

    def attach_saver(self, saver):
        """Initialize a new object with the saver.

        Parameters
        ----------
        saver: Save
            The saver.
        """
        new_ = copy.deepcopy(self)
        new_.saver = saver
        return new_


class CheckFunctor(CheckInterface, AttachSave, abc.ABC):
    """Functor to use with DefaultSave to check something about of a metadata item."""

    @abc.abstractmethod
    def __call__(self, metadata_item, feature_name, data_dict):
        """Check something about the metadata item.

        Parameters
        ----------
        metadata_item: Dict[str, Any]
            The metadata item. Will have keys :attr:`DefaultSave.FILENAME_STR`,
            :attr:`DefaultSave.FEATURE_NAME_STR`, :attr:`DefaultSave.SET_NAME_STR` and
            optionally :attr:`DefaultSave.OLD_FORMAT_STR`
        feature_name: Optional[str]
            The name of the feature.
        data_dict: Dict[str, Any]
            The data dict containing the data to save.

        Returns
        -------
        Union[Any, bool, None]
            False if the data does not pass the check, otherwise a useful representation
            of the data (such as the path). If None is returned, the check was not
            applicable.
        """
        pass


class IsDoneCheck(CheckFunctor):
    """Check if data has already been saved."""

    def __call__(self, metadata_item, feature_name, data_dict):
        """Check if data has already been saved.

        Parameters
        ----------
        metadata_item: Dict[str, Any]
            The metadata item. Will have keys :attr:`DefaultSave.FILENAME_STR`,
            :attr:`DefaultSave.FEATURE_NAME_STR`, :attr:`DefaultSave.SET_NAME_STR` and
            optionally :attr:`DefaultSave.OLD_FORMAT_STR`
        feature_name: Optional[str]
            The name of the feature.

        Returns
        -------
        Union[str, bool, None]
            False if the data has not been saved.
            None if the check was not applicable for the given feature name.
            The path to the data otherwise

        """
        if metadata_item[self.saver.metadata.FEATURE_NAME_STR] == feature_name:
            path = os.path.join(
                self.saver.root_dir, metadata_item[self.saver.metadata.FILENAME_STR]
            )
            return path if os.path.exists(path) else False


class IsReloadableCheck(CheckFunctor):
    """Check if data has already been saved and is reloadable."""

    def __call__(
        self,
        metadata_item: Dict[str, Any],
        feature_name: Optional[str],
        data_dict: Dict[str, Any],
    ):
        """Check if data has already been saved and is reloadable.

        Parameters
        ----------
        metadata_item: Dict[str, Any]
            The metadata item. Will have keys :attr:`DefaultSave.FILENAME_STR`,
            :attr:`DefaultSave.FEATURE_NAME_STR`, :attr:`DefaultSave.SET_NAME_STR` and
            optionally :attr:`DefaultSave.OLD_FORMAT_STR`.
        feature_name: Optional[str]
            The name of the feature.
        data_dict: Dict[str, Any]
            The data dict containing the unprocessed data

        Returns
        -------
        Union[str, bool, None]
            False if the data is not reloadable.
            None if the check was not applicable for the given feature name.
            The path to the data otherwise.

        """
        if metadata_item[self.saver.metadata.FEATURE_NAME_STR] is None:
            is_old_format = self.saver.metadata.is_old_format(metadata_item)
            filename = metadata_item[self.saver.metadata.FILENAME_STR]
            expected_filename = os.path.relpath(
                self.saver.filename_fn(data_dict, None, None), self.saver.root_dir
            )
            # If old format and not the expected filename to be reloadable
            # then skip
            if is_old_format and filename != expected_filename:
                return False

            path = os.path.join(
                self.saver.root_dir, metadata_item[self.saver.metadata.FILENAME_STR]
            )
            return path if os.path.exists(path) else False


class SaveMetadata(AttachSave, abc.ABC):
    """Abstract class for metadata to use when saving/reloading."""

    FEATURE_NAME_STR = "feature_name"
    FILENAME_STR = "filename"
    SET_NAME_STR = "set_name"
    OLD_FORMAT_STR = "old_format"

    def __init__(
        self, key_fn: Callable[[Dict[str, Any]], str] = default_metadata_key_fn
    ):
        """Create a new SaveMetadata.

        Parameters
        ----------
        key_fn: Callable[[Dict[str, Any]], str]
            The function to use to get the key from the data dict.
        """
        super().__init__()
        self.key_fn = key_fn

    @abc.abstractmethod
    def clear(self):
        """Clear the metadata."""

    @abc.abstractmethod
    def add(
        self,
        data_dict: Dict[str, Any],
        filepath: str,
        feature_name: Optional[str],
        set_name: Optional[str],
    ):
        """Add a metadata entry.

        Parameters
        ----------
        data_dict: Dict[str, Any]
            The data dict containing the data to save.
        filepath: str
            The path to the data.
        feature_name: Optional[str]
            The name of the feature.
        set_name: Optional[str]
            The name of the set.
        """

    @abc.abstractmethod
    def __contains__(self, item):
        """Check if the metadata contains a certain item.

        Parameters
        ----------
        item: Any
            The item to check.

        Returns
        -------
        bool
            Whether the item is contained.
        """

    @abc.abstractmethod
    def __getitem__(self, key: Any):
        """Retrieve a metadata item.

        Parameters
        ----------
        key: Any
            The key to retrieve.

        Returns
        -------
        Dict[str, Any]
            The metadata item.
        """

    @classmethod
    def is_old_format(cls, metadata_item: Union[str, Dict[str, Any]]):
        """Check if the metadata item is in the old format.

        Parameters
        ----------
        metadata_item: Union[str, Dict[str, Any]]
            The metadata item to check.

        Returns
        -------
        bool
            Whether the metadata item is in the old format.
        """
        return isinstance(metadata_item, str) or cls.OLD_FORMAT_STR in metadata_item


class OldMetadataCompliant(abc.ABC):
    """Mixin class for metadata that is compliant with the old format."""

    @abc.abstractmethod
    def convert_old_format(self, metadata_item: str, data_dict: Dict[str, Any]):
        """Convert the metadata item from the old format.

        Parameters
        ----------
        metadata_item: str
            The metadata item to convert.

        data_dict: Dict[str, Any]
            The data dict containing the data to save.

        Returns
        -------
        Dict[str, Any]
            The converted metadata item.
        """


class DefaultSaveMetadata(OldMetadataCompliant, SaveMetadata):
    """Implementation of SaveMetadata to work with DefaultSave."""

    def __init__(
        self,
        key_fn: Callable[[Dict[str, Any]], str] = default_metadata_key_fn,
        filename: str = ".save_metadata.json",
    ):
        """Create a new DefaultSaveMetadata.

        Parameters
        ----------
        key_fn: Callable[[Dict[str, Any]], str]
            The function to use to get the key from the data dict.
        filename: str
            The filename to use for the metadata.
        """
        super().__init__(key_fn=key_fn)
        self.filename = filename
        self.saver = None

    def get_path(self):
        """Get the path to the metadata file.

        Returns
        -------
        str
            The path to the metadata file.
        """
        if self.saver is not None and isinstance(self.saver, DefaultSave):
            path = os.path.join(self.saver.root_dir, self.filename)
            return path
        return self.filename

    def get_relpath(self, path: str):
        """Construct a relative path with regard to save folder.

        Parameters
        ----------
        path: str
            The path to make relative. If no saver is attached, this is returned as is.

        Returns
        -------
        str
            The relative path.
        """
        if (
            self.saver is not None
            and isinstance(self.saver, DefaultSave)
            and os.path.isabs(path)
        ):
            return os.path.relpath(path, self.saver.root_dir)
        return path

    @property
    def lock(self):
        """Retrieve the lock to use for the metadata file.

        Returns
        -------
        multiprocessing.Lock
            The lock to use for the metadata file.
        """
        return MultiprocessingSingleton.get_lock(self.get_path())

    def clear(self):
        """Clear the metadata."""
        self.lock.acquire()
        metadata_path = self.get_path()
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        self.lock.release()

    def get_metadata_for_savepath(
        self,
        path: str,
        feature_name: Optional[str],
        set_name: Optional[str],
        from_old_format=False,
    ):
        """Get the metadata associated for path where data is saved.

        Parameters
        ----------
        path: str
            The path to the data.
        feature_name: Optional[str]
            The name of the feature.
        set_name: Optional[str]
            The name of the set.
        from_old_format: bool
            Whether the metadata is in the old format.

        Returns
        -------
        Dict[str, Any]
            The metadata associated with the path.
        """
        metadata = {
            self.FILENAME_STR: self.get_relpath(path),
            self.FEATURE_NAME_STR: feature_name,
            self.SET_NAME_STR: set_name,
        }
        if from_old_format:
            metadata[self.OLD_FORMAT_STR] = True
        return metadata

    def convert_old_format(self, metadata_item: str, data_dict: Dict[str, Any]):
        """Convert the metadata item from the old format.

        Parameters
        ----------
        metadata_item: str
            The metadata item to convert.
        data_dict: Dict[str, Any]
            The data dict containing the data to save.

        Returns
        -------
        Dict[str, Any]
            The converted metadata item.
        """
        return self.get_metadata_for_savepath(metadata_item, None, None, True)

    def get(self):
        """Load the metadata.

        Returns
        -------
        Dict[str, Any]
            The metadata.
        """
        metadata_path = self.get_path()
        if not os.path.exists(metadata_path):
            return {}
        self.lock.acquire()
        with open(metadata_path) as fp:
            metadata = json.load(fp)
        self.lock.release()
        return metadata

    def add(
        self,
        data_dict: Dict[str, Any],
        filepath: str,
        feature_name: Optional[str],
        set_name: Optional[str],
    ):
        """Add metadata for a file.

        Parameters
        ----------
        data_dict: Dict[str, Any]
            The data dictionary.
        filepath: str
            The path to the file.
        feature_name: Optional[str]
            The name of the feature.
        set_name: Optional[str]
            The name of the set.
        """
        metadata = self.get()
        key = self.key_fn(data_dict)
        if key not in metadata:
            metadata[key] = []
        all_filepaths = wrap_in_list(filepath)
        for path in all_filepaths:
            metadata_for_savepath = self.get_metadata_for_savepath(
                path, feature_name, set_name
            )
            if metadata_for_savepath not in metadata[key]:
                metadata[key] += [metadata_for_savepath]
        self.write(metadata)

    def write(self, metadata_dict: Dict[str, Any]):
        """Write the metadata to disk.

        Parameters
        ----------
        metadata_dict: Dict[str, Any]
            A dictionary containing the metadata.
        """
        self.lock.acquire()
        with open(self.get_path(), "w") as fp:
            json.dump(metadata_dict, fp)
        self.lock.release()

    def __contains__(self, item: Any):
        """Check if the metadata contains a certain item.

        Parameters
        ----------
        item: Any

        Returns
        -------
        bool
            Whether the item is contained.
        """
        return item in self.get()

    def __getitem__(self, key: Any):
        """Retrieve a metadata item.

        Parameters
        ----------
        key: Any
            The key to retrieve.

        Returns
        -------
        Any
            The metadata item.
        """
        return self.get()[key]


class DefaultSave(Save):
    """Default save class.

    This class will save data_dicts to disk, but also keep a metadata file
    (:attr:`Save.metadata_filename`) that contains the information about the mapping
    between an unprocessed input filename and multiple possible output filenames.
    """

    DEFAULT_SAVE_FUNCTIONS = {
        "npy": np.save,
        "pickle": pickle_dump_wrapper,
        "data_dict": pickle_dump_wrapper,
    }
    DEFAULT_RELOAD_FUNCTIONS = {
        "npy": np.load,
        "npz": np.load,
        "pickle": pickle_load_wrapper,
        "data_dict": pickle_load_wrapper,
    }

    _metadata_deprecation_warning_logged = False

    def __init__(
        self,
        root_dir: str,
        to_save: Optional[Mapping[str, Any]] = None,
        overwrite: bool = False,
        clear_output: bool = False,
        filename_fn: FilenameFnInterface = DefaultFilenameFn(),
        metadata: SaveMetadata = DefaultSaveMetadata(),
        save_fn: SaveFnInterface = None,
        reload_fn: ReloadFnInterface = None,
        check_done: Optional[CheckInterface] = IsDoneCheck(),
        check_reloadable: Optional[CheckInterface] = IsReloadableCheck(),
    ):
        """Create a Save step.

        Parameters
        ----------
        root_dir: str
            The root directory where the data should be saved.
        to_save: Optional[Mapping[str, Any]]
            The data to save. If None, the data_dict is saved entirely. If a mapping
            between feature names and the key of the data in the data_dict is given,
            only the data for the provided features is saved.
        overwrite: bool
            Whether to overwrite existing files.
        clear_output: bool
            Whether to clear the output data_dict after saving. This can save space
            when save is the last step in a pipeline.
        filename_fn: FilenameFnInterface
            A function to generate a filename for the data. The function should take
            the data_dict, the feature name, the set name and a separator as input
            and return a filename.
        save_fn: SaveFnInterface
            A function to save the data. The function should take the data and the
            filepath as inputs and save the data. If a mapping between file extensions
            and functions is given, the function corresponding to the file extension
            is used to save the data. If None, the default save functions (defined in
            self.DEFAULT_SAVE_FUNCTIONS) are used.
        reload_fn: ReloadFnInterface
            A function to reload the data. The function should take the filepath as
            input and return the data. If a mapping between file extensions and
            functions is given, the function corresponding to the file extension is
            used to reload the data. If None, the default reload functions (defined in
            self.DEFAULT_RELOAD_FUNCTIONS) are used.
        check_done: Optional[CheckInterface]
            A functor to check whether the data has already been saved. If None, no
            checking is done
        check_reloadable: Optional[CheckInterface]
            A functor to check whether the data can be reloaded. if None, no checking is
            done.
        """
        super().__init__(clear_output=clear_output, overwrite=overwrite)
        self.root_dir = root_dir
        self.to_save = to_save
        self.filename_fn = filename_fn
        self.save_fn = save_fn
        if self.save_fn is None:
            self.save_fn = self.DEFAULT_SAVE_FUNCTIONS
        self.reload_fn = reload_fn
        if self.reload_fn is None:
            self.reload_fn = self.DEFAULT_RELOAD_FUNCTIONS
        self.check_done = self._attach_saver(check_done)
        self.check_reloadable = self._attach_saver(check_reloadable)
        self.metadata = self._attach_saver(metadata)

    def _attach_saver(self, check: Optional[AttachSave]):
        """Attach this :class:`Save` object to a :class:`AttachSave` object.

        Parameters
        ----------
        check: Optional[AttachSave]
            An optional object to attach this :class:`Save` object to.
            Note that the object should implement the :meth:`attach_saver` method, which
            will return a copy of the :class:`AttachSave` object with the :class:`Save`
            object attached.

        Returns
        -------
        Optional[CheckInterface]
            The prepared check.
        """
        if hasattr(check, "attach_saver"):
            return check.attach_saver(self)
        return check

    @property
    def overwrite(self):
        """Whether to overwrite existing files.

        Returns
        -------
        bool
            Whether to overwrite existing files.
        """
        return self._overwrite

    @overwrite.setter
    def overwrite(self, value):
        """Set whether to overwrite existing files.

        Parameters
        ----------
        value: bool
            Whether to overwrite existing files.
        """
        self._overwrite = value
        # Clear the metadata if overwrite is True and it has been initialized
        if self._overwrite and hasattr(self, "metadata"):
            self.metadata.clear()

    def is_already_done(self, data_dict):
        """Check whether the data_dict has already been saved.

        Parameters
        ----------
        data_dict: Dict[str, Any]
            The data_dict to check.

        Returns
        -------
        bool
            Whether the data_dict has already been saved. This will be checked in the
            stored metadata.
        """
        # If overwrite is True, the data is never done
        if self.overwrite:
            return False
        # If the key is not in the metadata, the data is not done
        key = self.metadata.key_fn(data_dict)
        if key not in self.metadata:
            return False
        # If the key is in the metadata, check whether the data is done
        results = self._iterate_over_metadata_item(
            self.metadata[key], self.check_done, data_dict
        )
        # All items should be done for it to be considered done
        return len(results) and all(results)

    def _iterate_over_metadata_item(self, metadata_item, callback, data_dict):
        # Make a list of metadata items
        file_infos = wrap_in_list(metadata_item)
        # Select the feature names for this Save step
        feature_names = [None] if self.to_save is None else self.to_save.keys()
        # Iterate over all file_infos to check them
        results = []
        for info in file_infos:
            # Info should be a dict, but it can also be a string if the data was saved
            # with the old metadata format (version <= 0.0.2).
            if self.metadata.is_old_format(info):
                # Only log this if it hasn't been logged before.
                if not self._metadata_deprecation_warning_logged:
                    logging.warning(
                        "Found previously saved data with the old metadata format "
                        "(version <= 0.0.2). DefaultSave will attempt to reload the "
                        "data, but it is recommended to delete the old data and "
                        "metadata file if possible."
                    )
                    self._metadata_deprecation_warning_logged = True
                if isinstance(self.metadata, OldMetadataCompliant):
                    info = self.metadata.convert_old_format(info, data_dict)
                else:
                    raise ValueError(
                        "The metadata class used is not compatible with the old "
                        "format. Please delete the old data and metadata file."
                    )

            for feature_name in feature_names:
                item = callback(info, feature_name, data_dict)
                # If the callback returns None, the check is skipped because it is
                # assumed that the check is not applicable.
                if item is not None:
                    results.append(item)
        return results

    def _serialization_wrapper(self, fn, filepath, *args, action="save", **kwargs):
        if not isinstance(fn, dict):
            return fn(filepath, *args, **kwargs)
        suffix = os.path.basename(filepath).split(".")[-1]
        if suffix not in fn:
            raise ValueError(
                f"Can't find an appropriate function to {action} '{filepath}'."
            )
        directory = os.path.dirname(filepath)
        os.makedirs(directory, exist_ok=True)
        return fn[suffix](filepath, *args, **kwargs)

    def _apply_to_data(self, data_dict, fn):
        # Save in .data_dict
        if self.to_save is None:
            path = os.path.join(self.root_dir, self.filename_fn(data_dict, None, None))
            self._serialization_wrapper(fn, path, data_dict, action="save")
            self.metadata.add(data_dict, [path], None, None)
            return

        # Save singular features
        paths = []
        for feature_name, feature_loc in self.to_save.items():
            data = data_dict[feature_loc]
            # Per set
            if isinstance(data, dict):
                for set_name, set_data in data.items():
                    filename = self.filename_fn(data_dict, feature_name, set_name)
                    path = os.path.join(self.root_dir, filename)
                    self._serialization_wrapper(fn, path, set_data, action="save")
                    paths += [path]
                    self.metadata.add(data_dict, paths, feature_name, set_name)

            # In full
            else:
                filename = self.filename_fn(data_dict, feature_name, None)
                path = os.path.join(self.root_dir, filename)
                self._serialization_wrapper(fn, path, data, action="save")
                paths += [path]
                self.metadata.add(data_dict, paths, feature_name, None)

    def is_reloadable(self, data_dict: Dict[str, Any]) -> bool:
        """Check whether an already processed data_dict can be reloaded.

        Parameters
        ----------
        data_dict: Dict[str, Any]
            The data_dict for which we want to reload the already processed version.

        Returns
        -------
        bool
            Whether an already processed data_dict can be reloaded to continue
            processing.
        """
        key = self.metadata.key_fn(data_dict)
        if key not in self.metadata:
            return False

        # No support to reload singular features
        if self.to_save is not None:
            return False

        # If overwrite is True, no reloading is possible
        if self.overwrite:
            return False

        is_reloadable = self._iterate_over_metadata_item(
            self.metadata[key], self.check_reloadable, data_dict
        )
        return len(is_reloadable) and any(is_reloadable)

    def reload(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Reload the data_dict from the saved file.

        Parameters
        ----------
        data_dict: Dict[str, Any]
            The data_dict for which we want to reload the already processed version.

        Returns
        -------
        Dict[str, Any]
            The reloaded data_dict.
        """
        key = self.metadata.key_fn(data_dict)
        paths = self._iterate_over_metadata_item(
            self.metadata[key], self.check_reloadable, data_dict
        )
        if not (len(paths) and any(paths)):
            raise ValueError("Didn't find any file that can be reloaded.")

        selected_path = [p for p in paths if p][0]
        return self._serialization_wrapper(
            self.reload_fn,
            selected_path,
            action="reload",
        )

    def __call__(self, data_dict):
        """Save the data_dict to the :attr:`root_dir`.

        Parameters
        ----------
        data_dict: Dict[str, Any]
            The data_dict to save.

        Returns
        -------
        Dict[str, Any]
            The data_dict if :attr:`clear_output` is False, an empty dict otherwise.
        """
        os.makedirs(self.root_dir, exist_ok=True)
        self._apply_to_data(data_dict, self.save_fn)
        # Save some RAM space
        if self.clear_output:
            # Explicitly clean up
            gc.collect()
            return {}
        return data_dict
