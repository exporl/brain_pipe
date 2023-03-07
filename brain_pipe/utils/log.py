"""Logging utilities."""
import logging


class DefaultFormatter(logging.Formatter):
    """Default formatter for the brain_pipe package."""

    default_format = "%(asctime)s | %(process)-3d | %(levelname)-8s | %(message)s"

    def __init__(self, *args, fmt=default_format, **kwargs):
        """Create a new DefaultFormatter instance.

        Parameters
        ----------
        args: Any
            Additional positional arguments to be passed to logging.Formatter.
        fmt: str
            The format string to be used.
        kwargs: Dict[str, Any]
            Additional keyword arguments to be passed to logging.Formatter.
        """
        super().__init__(*args, fmt=fmt, **kwargs)


class ColorFormatter(DefaultFormatter):
    """Color formatter for the brain_pipe package."""

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    COLOR_MAPPING = {
        logging.DEBUG: grey,
        logging.INFO: grey,
        logging.WARNING: yellow,
        logging.ERROR: red,
        logging.CRITICAL: bold_red,
    }

    def format(self, record):  # noqa: D102
        output = super(ColorFormatter, self).format(record)
        return self.COLOR_MAPPING[record.levelno] + output + self.reset


def default_logging(**kwargs):
    """Set the default logging configuration.

    Parameters
    ----------
    kwargs: Dict[str, Any]
        Additional keyword arguments to be passed to logging.basicConfig.

    Returns
    -------
    logging.basicConfig
        The logging configuration.
    """
    # Remove previous logging handlers
    root = logging.getLogger()
    for handler in root.handlers:
        root.removeHandler(handler)
    # Create a StreamHandler with a default ColorFormatter
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(ColorFormatter())
    extra_handlers = kwargs.get("handlers", [])
    root.setLevel(kwargs.get("level", logging.INFO))
    [root.addHandler(hand) for hand in [ch] + extra_handlers]
    logging.root = root
