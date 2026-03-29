"""
Shared utilities: logging setup, serialization helpers.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger with console handler.

    Args:
        name: Logger name (typically ``__name__``).
        level: Logging level.

    Returns:
        Configured :class:`logging.Logger`.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def save_pickle(obj: Any, path: Path) -> None:
    """Pickle *obj* to *path*, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Path) -> Any:
    """Load and return the object stored in the pickle file at *path*."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(obj: Any, path: Path) -> None:
    """Write *obj* as pretty-printed JSON to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Any:
    """Read and return JSON from *path*."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
