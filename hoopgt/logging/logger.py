import logging
from typing import Any

from colorama import Fore, Style, init

from hoopgt.logging.filter import apply_warning_filter, remove_warning_filter

init(autoreset=True)


class HoopGTLoggerContext:
    """
    Context to manage the hoopgt_logger logging level and warning filters.

    Parameters
    ----------
    verbose : bool
        Whether to log at high detail or not.
    logging_level : int
        The logging level to set for the hoopgt_logger.
    """

    active_contexts: list["HoopGTLoggerContext"] = []

    def __init__(self, verbose: bool, logging_level: int = logging.INFO) -> None:
        self.verbose = verbose
        self.original_level = 0
        self.logging_level = logging_level

    def __enter__(self) -> None:
        """Enter the context manager."""
        self.original_level = hoopgt_logger.getEffectiveLevel()
        self.active_contexts.append(self)

        if not self.verbose:
            apply_warning_filter()

        # Only set the level if it would make logging more restrictive
        new_level = logging.DEBUG if self.verbose else self.logging_level
        if new_level > self.original_level or len(self.active_contexts) == 1:
            hoopgt_logger.setLevel(new_level)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        Exit the context manager.

        Parameters
        ----------
        exc_type : Any
            The type of the exception.
        exc_val : Any
            The value of the exception.
        exc_tb : Any
            The traceback of the exception.
        """
        hoopgt_logger.setLevel(self.original_level)
        # if no more contexts are active, remove the warning filter
        if len(self.active_contexts) == 1:
            remove_warning_filter()
        self.active_contexts.remove(self)


class CustomFormatter(logging.Formatter):
    """
    Custom formatter for logging with support for different formats and styles.

    Parameters
    ----------
    fmt : str
        The format string for the log messages.
    datefmt : str
        The format string for the date in log messages.
    style : str
        The style of the format string ('%', '{', or '$').
    validate : bool
        Whether to validate the format string.
    defaults : dict, optional
        A dictionary of default values for the formatter.

    Attributes
    ----------
    COLORS : dict
        The colors for the log levels.
    """

    COLORS = {
        "DEBUG": Fore.CYAN,
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.MAGENTA,
    }

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record with colors based on the log level.

        Parameters
        ----------
        record : logging.LogRecord
            The log record.

        Returns
        -------
        str
            The formatted log message.
        """
        log_message = super().format(record)
        log_level = record.levelname

        if log_level in self.COLORS:
            color = self.COLORS[log_level]
            return f"{color}{log_message}{Style.RESET_ALL}"
        else:
            return log_message


def setup_hoopgt_logger() -> logging.Logger:
    """
    Set up the hoopgt_logger with a custom formatter that adds colors based on log level.

    Returns
    -------
    logging.Logger
        The hoopgt_logger.
    """
    hoopgt_logger = logging.getLogger("hoopgt_logger")
    hoopgt_logger.setLevel(logging.INFO)

    if not hoopgt_logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(CustomFormatter("%(levelname)s - %(message)s"))
        hoopgt_logger.addHandler(console_handler)

    # avoid duplicate logging messages
    hoopgt_logger.propagate = False

    return hoopgt_logger


hoopgt_logger = setup_hoopgt_logger()
