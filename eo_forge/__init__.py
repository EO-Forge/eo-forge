"""
EO-Forge package
"""

# Set the version number from VERSION file generated during the package installation
from pathlib import Path

PACKAGE_ROOT_DIR = Path(__file__).parent

with open(PACKAGE_ROOT_DIR / "VERSION", "r") as version_file:
    __version__ = version_file.readline()
    version = __version__

import sys

import logging

default_logger = None


def check_logger(logger=None):
    """
    Check that always return a logger.
    If logger=None, the default library's logger is returned.
    Otherwise, the `logger` argument is returned.
    """
    if (logger is not None) and (not isinstance(logger, logging.Logger)):
        raise RuntimeError("None or logging instance are valid values for logger")

    if logger is None:
        return default_logger

    return logger


def set_default_logger(logger=None):
    """
    Set the default logger to be used by the library.
    By default, a logger printing to the stdout is used.

    The name of the default logger is "eo_forge.default".
    """
    global default_logger

    if logger is None:
        # Define a default logger that print all the messages to the stdout
        default_logger = logging.getLogger(f"{__name__}.default")

        # If a root logger is defined, then we use that one.
        # This check can be done by checking if the recently create logger has
        # any handler. When the root logger is not defined, the created logger has
        # no handlers.
        if not logging.getLogger().hasHandlers():
            # Not root logger. Let's create one.
            default_logger.setLevel(logging.DEBUG)
            handler = logging.StreamHandler(sys.stdout)

            fmt = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y/%m/%d %H:%M:%S"
            )

            handler.setFormatter(fmt)

            default_logger.addHandler(handler)
    else:
        # Use the provided logger.
        default_logger = logger


def set_default_logger_level(level):
    global default_logger
    default_logger.setLevel(level)  # noqa


set_default_logger()
