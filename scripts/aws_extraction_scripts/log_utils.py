"""
log_utils.py
------------
Unified logger for IntelliDoc / Doc-Understand OCR pipeline.

Writes logs to the LOG_DIR defined in config.py, e.g.:

  logs/aws_extraction_logs/<stage>.log

and also streams output to stdout so Airflow UI / local console
can see the logs.
"""

import os
import sys
import logging
import atexit
from pathlib import Path

# ---------------------------------------------------------------------
# 🔧 Resolve project root & import config.LOG_DIR
# ---------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.aws_extraction_scripts.config import LOG_DIR as CONFIG_LOG_DIR

# Use the LOG_DIR coming from config.py (handles Airflow vs local)
LOG_DIR = Path(CONFIG_LOG_DIR)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# 🧠 Logger Setup Function
# ---------------------------------------------------------------------
def get_logger(stage: str) -> logging.Logger:
    """
    Returns a dual logger (file + stdout).

    - File logs go to: LOG_DIR / "<stage>.log"
    - Console logs go to stdout (captured by Airflow / terminal)

    This function is safe to call many times for the same stage;
    it clears old handlers to avoid duplicate messages.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{stage}.log"

    logger = logging.getLogger(stage)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if logger is reused
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    # ✅ File handler (persistent logs)
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)

    # ✅ Console handler (for Airflow UI / local terminal)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    ch.setLevel(logging.INFO)

    # Attach handlers
    logger.addHandler(fh)
    logger.addHandler(ch)

    # Allow messages to propagate to root, so Airflow / other handlers see them
    logger.propagate = True

    # ✅ Ensure root logger also has at least the console handler
    root = logging.getLogger()
    if not root.hasHandlers():
        root.addHandler(ch)
    root.setLevel(logging.INFO)

    # Clean up on exit
    atexit.register(lambda: [h.close() for h in logger.handlers])

    return logger
