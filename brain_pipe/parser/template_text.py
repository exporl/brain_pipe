"""Class that can parse text with template stings into Runners and Pipelines."""
import abc
from typing import Dict, Any

import jinja2
import jinja2.meta
from brain_pipe.parser.text import TextParser


class TemplateTextParser(TextParser, abc.ABC):
    """Parser that can parse text with template stings.

    This parser uses jinja2 to fill in templates in the text.
    The rendered and filled-in text is then parsed by the TextParser.
    """

    DEFAULT_ENVIRONMENT_KWARGS = {
        "autoescape": True,
        "undefined": jinja2.StrictUndefined,
    }

    def __init__(self):
        """Create a new TemplateTextParser."""
        super().__init__()
        self.environment = jinja2.Environment(**self.DEFAULT_ENVIRONMENT_KWARGS)

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
        rendered_text = self.environment.from_string(text).render()
        return super(TemplateTextParser, self).__call__(rendered_text)

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
        args_dict = super().get_additional_kwargs(input_, info)
        ast = self.environment.parse(input_)
        for undeclared_var in jinja2.meta.find_undeclared_variables(ast):
            args_dict[undeclared_var] = {
                "help": f"Variable '{undeclared_var}', as used in the template",
                "required": True,
            }
        return args_dict

    def set_additional_kwargs(self, kwargs: Dict[str, Any]):
        """Set additional arguments for the parser.

        Parameters
        ----------
        kwargs: Dict[str, Any]
            A dictionary containing the names of the arguments as keys and the options
            (kwargs) for a (CLI) argument parser as values.
        """
        super().set_additional_kwargs(kwargs)
        env_kwargs = self.DEFAULT_ENVIRONMENT_KWARGS.copy()
        jinja2_kwargs = {
            key: value for key, value in kwargs.items() if key.startswith("jinja2_")
        }
        if len(jinja2_kwargs):
            env_kwargs.update(jinja2_kwargs)
            self.environment = jinja2.Environment(**env_kwargs)
        for arg_name, arg_value in kwargs.items():
            self.environment.globals[arg_name] = arg_value
