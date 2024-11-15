# log_config.py
"""
This module generates log files for monitoring.
"""
import logging
from logging.handlers import RotatingFileHandler
import os
import time
from functools import wraps

logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger()

def log_event(message, level="info"):
    """
    Log events to the log file.
    Args:
        message (str): Message to log.
        level (str): Log level (info, warning, error, critical).
    """
    if level == "info":
        logger.info(message)
    elif level == "warning":
        logger.warning(message)
    elif level == "error":
        logger.error(message)
    elif level == "critical":
        logger.critical(message)

def log_execution_time(func):
    """
    Decorator to log the execution time of a function.
    Args:
        func (function): Function to be wrapped.
    Returns:
        function: Wrapped function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logger.info(f"Function '{func.__name__}' executed in {elapsed_time:.2f} seconds.")
        return result

    return wrapper

# Use RotatingFileHandler to limit log file size
handler = RotatingFileHandler(
    "logs/jats.log", maxBytes=5_000_000, backupCount=5  # 5 MB max size, 5 backups
)
logger.addHandler(handler)