"""Structured logging system for the Aura project.

Provides custom ANSI colored console formatting, synchronized file logging,
and configurable log handlers across all modules.
"""

import logging
from pathlib import Path
import sys
from typing import Optional

from src.utils.paths import project_paths


class ColoredConsoleFormatter(logging.Formatter):
    """Custom Formatter adding ANSI colors to console log levels."""

    GREY = "\x1b[38;20m"
    BLUE = "\x1b[34;20m"
    GREEN = "\x1b[32;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"

    FMT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
    DATE_FMT = "%Y-%m-%d %H:%M:%S"

    FORMATS = {
        logging.DEBUG: GREY + FMT + RESET,
        logging.INFO: GREEN + FMT + RESET,
        logging.WARNING: YELLOW + FMT + RESET,
        logging.ERROR: RED + FMT + RESET,
        logging.CRITICAL: BOLD_RED + FMT + RESET,
    }

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno, self.FMT)
        formatter = logging.Formatter(log_fmt, datefmt=self.DATE_FMT)
        return formatter.format(record)


def setup_logger(
    name: str = "aura",
    level: str | int = "INFO",
    log_file: Optional[Path | str] = None,
    log_to_console: bool = True,
) -> logging.Logger:
    """Configures and returns a structured Logger instance.

    Args:
        name: Root module name for logger.
        level: Logging level threshold ("DEBUG", "INFO", "WARNING", "ERROR", etc.).
        log_file: Optional path to output log file. If provided, appends plain logs.
        log_to_console: If True, attaches colored console handler.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if isinstance(level, str):
        numeric_level = getattr(logging, level.upper(), logging.INFO)
    else:
        numeric_level = level

    logger.setLevel(numeric_level)

    # Avoid duplicate handlers if setup_logger is called repeatedly
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console Handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(ColoredConsoleFormatter())
        logger.addHandler(console_handler)

    # File Handler
    if log_file is not None:
        file_path = Path(log_file).resolve()
        project_paths.ensure_dir(file_path.parent)

        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Prevent message duplication up to Python root logger
    logger.propagate = False

    return logger
