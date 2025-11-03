import logging
import sys
from pathlib import Path


def setup_logger(to_file: bool, log_dir: Path) -> logging.Logger:
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if to_file:
        file_handler = logging.FileHandler(log_dir / "train.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
