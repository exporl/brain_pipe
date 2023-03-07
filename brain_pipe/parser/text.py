"""Class that can parse text into Runners and Pipelines."""
import abc
from brain_pipe.parser.simple_dict import SimpleDictParser


class TextParser(SimpleDictParser, abc.ABC):
    """Parser based on SimpleDictParser that can parse text instead of dictionaries."""

    @abc.abstractmethod
    def parse_text_to_dict(self, text: str):
        """Parse text to a dictionary containing simple builtin types (str, int,...).

        Parameters
        ----------
        text: str
            The text to parse.

        Returns
        -------
        Dict[str, Any]
            The parsed dictionary.
        """
        pass

    def __call__(self, text: str):
        """Parse text to obtain a Runner and Dataloaders and Pipelines.

        Parameters
        ----------
        text: str
            The text to parse.

        Returns
        -------
        Tuple[Runner, List[Tuple[Pipeline, DataLoader]]]
            A runner and a list of tuples containing a Pipeline and its DataLoader.
        """
        text_dict = self.parse_text_to_dict(text)
        return super().__call__(text_dict)
