"""Logging Module.

WHY THIS MODULE EXISTS:
    Centralized logging framework for standardizing output formatting across console streams
    and persistent file logs.

HOW FUTURE MODULES WILL PLUG IN:
    - All submodules initialize standard Python loggers via `setup_logger()` or `logging.getLogger(__name__)`.
"""

from src.logging.logger import ColoredConsoleFormatter, setup_logger

__all__ = ["setup_logger", "ColoredConsoleFormatter"]
