import logging
import time
from typing import Any, Dict

from .config import settings

_logger = None


def get_logger() -> logging.Logger:
    global _logger
    if _logger:
        return _logger

    logger = logging.getLogger(settings.APP_NAME)
    logger.setLevel(settings.LOG_LEVEL.upper())

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s"
    ))
    logger.addHandler(handler)

    _logger = logger
    return logger


log = get_logger()


def timed(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            log.debug(f"{func.__name__} took {time.perf_counter()-start:.3f}s")
    return wrapper


def truncate_context(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[:max_chars] + "\n...[truncated]..."

