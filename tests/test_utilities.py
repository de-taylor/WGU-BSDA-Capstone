"""PyTest Unit Testing for the src.utilities module."""

# PyTest
import pytest
# Python Standard Library Modules
import logging

# Custom modules
from src.utilities import new_logger, save_atomic

# Unit Tests: src.utilities.new_logger
def test_new_logger_creates_logger(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    logger = new_logger("test_logger", str(log_dir))

    assert isinstance(logger, logging.Logger)  # verify Logger object
    assert logger.name == "test_logger"  # verify Logger object initialized correctly
    # verify handlers were instantiated correctly
    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)

# Unit Tests: src.utilities.save_atomic
def test_save_atomic(tmp_path):
    pass