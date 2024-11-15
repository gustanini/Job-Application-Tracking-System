# log_config.py
"""
This module generates log files for monitoring.
"""
import logging
import os

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