"""Parsers for YAML files."""
import yaml

from brain_pipe.parser.file import TemplateFileParser, FileParser
from brain_pipe.parser.text import TextParser


class YAMLMixin:
    """Mixin class for a TextParser classes to parse YAML text."""

    def parse_text_to_dict(self, text):
        """Parse YAML text to a dictionary.

        Parameters
        ----------
        text: str
            The YAML text to parse.

        Returns
        -------
        Dict[str, Any]
            The parsed dictionary.
        """
        return yaml.safe_load(text)


class YAMLTextParser(YAMLMixin, TextParser):
    """Parser based on TextParser that can parse YAML text."""

    pass


class YAMLFileParser(YAMLMixin, FileParser):
    """YAML parser that can parse YAML files."""

    pass


class YAMLTemplateFileParser(YAMLMixin, TemplateFileParser):
    """YAML parser that can parse YAML template files."""

    pass
