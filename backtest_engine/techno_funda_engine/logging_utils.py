"""
logging_utils.py
-----------------
Centralised logger so every module logs consistently to both console and a
rotating file under logs/. Import get_logger(__name__) anywhere.
"""
import logging
import os
from logging.handlers import RotatingFileHandler

from . import config


def get_logger(name: str) -> logging.Logger:
    os.makedirs(config.LOG_DIR, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:          # avoid duplicate handlers on re-import
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    file_handler = RotatingFileHandler(
        os.path.join(config.LOG_DIR, "engine.log"),
        maxBytes=5_000_000,
        backupCount=3,
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger
