"""Parser based on TextDictParser that can parse files instead of dictionaries."""
import abc
import os
from typing import Any, Dict

from brain_pipe.parser.template_text import TemplateTextParser
from brain_pipe.parser.text import TextParser


class FileLoadMixin:
    """Mixin class for a TextParser to load a file before parsing."""

    READ_MODE = "r"

    def load_file(self, path: str) -> str:
        """Load a file into a text that can be parsed.

        Parameters
        ----------
        path: str
            The path to the file to load.

        Returns
        -------
        str
            The text that was loaded from the file.
        """
        with open(path, self.READ_MODE) as f:
            return f.read()

    def __call__(self, path):
        """Parse the text dictionary.

        Parameters
        ----------
        path: str
            The text dictionary to parse.

        Returns
        -------
        Tuple[Runner, List[Tuple[Pipeline, DataLoader]]]
            A runner and a list of tuples containing a Pipeline and its DataLoader.
        """
        contents = self.load_file(path)
        return super().__call__(contents)

    def get_additional_kwargs(
        self, input_: Any, info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Find the additional arguments that should be passed to the parser.

        This function identifies additional arguments that should be passed to the
        parser by e.g. a CLI.

        Parameters
        ----------
        input_: str
            The input string that should be parsed when the parser is called.

        info: Dict[str, Any]
            A dictionary containing available info, such as e.g. parsed CLI arguments.
            This information can be used to determine the additional arguments.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the names of the arguments as keys and the options
            (kwargs) for a (CLI) argument parser as values.
        """
        new_input = self.load_file(input_)
        return super().get_additional_kwargs(new_input, info)


class FileParser(FileLoadMixin, TextParser, abc.ABC):
    """Parser based on TextParser that can parse files instead of text."""

    pass


class TemplateFileParser(FileLoadMixin, TemplateTextParser, abc.ABC):
    """Parser based on TemplateTextParser that can parse files instead of text."""

    FILEDIR_STR = "__filedir__"
    FILE_STR = "__file__"

    def get_additional_kwargs(
        self, input_: Any, info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Find the additional arguments that should be passed to the parser.

        This function identifies additional arguments that should be passed to the
        parser by e.g. a CLI.

        Parameters
        ----------
        input_: str
            The input string that should be parsed when the parser is called.

        info: Dict[str, Any]
            A dictionary containing available info, such as e.g. parsed CLI arguments.
            This information can be used to determine the additional arguments.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the names of the arguments as keys and the options
            (kwargs) for a (CLI) argument parser as values.
        """
        result = super().get_additional_kwargs(input_, info)
        abspath = os.path.abspath(input_)
        if self.FILEDIR_STR in result:
            self.environment.globals[self.FILEDIR_STR] = os.path.dirname(abspath)
            del result[self.FILEDIR_STR]
        if self.FILE_STR in result:
            self.environment.globals[self.FILE_STR] = abspath
            del result[self.FILE_STR]
        return result
