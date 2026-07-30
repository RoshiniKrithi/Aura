"""Unit tests for logging system."""

import logging

from src.logging.logger import setup_logger


def test_setup_logger_console_and_file(temp_dir):
    """Verify logger setup with console and file handler."""
    log_path = temp_dir / "test_run.log"
    logger = setup_logger(name="test_aura", level="DEBUG", log_file=log_path)

    assert logger.level == logging.DEBUG
    assert len(logger.handlers) == 2

    logger.info("Test log entry")

    assert log_path.exists()
    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "Test log entry" in content

    # Close handlers so Windows temporary directory cleanup can remove log_path
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

