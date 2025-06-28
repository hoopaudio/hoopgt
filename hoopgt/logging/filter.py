import os
import sys
import warnings
from typing import Any


def apply_warning_filter() -> None:
    """Apply the warning filter globally."""
    warnings.filterwarnings("ignore")


def remove_warning_filter() -> None:
    """Remove the warning filter globally."""
    warnings.filterwarnings("default")


def is_notebook() -> bool:
    """
    Check if the code is running in a Jupyter notebook.

    Returns
    -------
    bool
        True if the code is running in a Jupyter notebook, False otherwise.
    """
    try:
        from IPython.core.getipython import get_ipython

        shell = get_ipython().__class__.__name__
        # Jupyter notebook or qtconsole result in True, terminal result in False
        return shell == "ZMQInteractiveShell"
    except NameError:
        return False  # Probably standard Python interpreter


class SuppressOutput:
    """Context manager to suppress output in console or Jupyter notebook."""

    def __enter__(self) -> "SuppressOutput":
        """Enter the context manager."""
        self._is_notebook = is_notebook()
        # Use universal approach that works in both notebook and terminal
        import io
        
        self._original_stdout = sys.stdout
        self._buffer = io.StringIO()
        sys.stdout = self._buffer
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Exit the context manager."""
        # Restore original stdout for both notebook and terminal
        sys.stdout = self._original_stdout
