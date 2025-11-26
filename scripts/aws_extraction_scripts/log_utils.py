"""
log_utils.py
------------
Unified logger for IntelliDoc OCR pipeline.
Writes logs to:
  logs/aws_extraction_logs/<stage>.log
and streams output to Airflow UI.
"""

import os
import sys
import logging
import atexit

# ---------------------------------------------------------------------
# 🔧 Resolve project root for both Docker & local
# ---------------------------------------------------------------------
if os.path.exists("/opt/project"):
    PROJECT_ROOT = "/opt/project"
else:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# ---------------------------------------------------------------------
# 📁 Centralized log directory
# ---------------------------------------------------------------------
LOG_DIR = os.path.join(PROJECT_ROOT, "logs", "aws_extraction_logs")
os.makedirs(LOG_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# 🧠 Logger Setup Function
# ---------------------------------------------------------------------
def get_logger(stage: str):
    """
    Returns a dual logger (Airflow + file).
    Ensures both Airflow UI and local logs get all messages.
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"{stage}.log")

    logger = logging.getLogger(stage)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )

    # ✅ File handler (persistent logs)
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)

    # ✅ Console handler (for Airflow UI)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    ch.setLevel(logging.INFO)

    # Attach handlers
    logger.addHandler(fh)
    logger.addHandler(ch)

    # ✅ Critical line: allow Airflow to capture stdout + keep file logs
    logger.propagate = True

    # ✅ Sync Python's root stdout so `print()` also goes into Airflow logs
    root = logging.getLogger()
    if not root.hasHandlers():
        root.addHandler(ch)
    root.setLevel(logging.INFO)

    # Clean up on exit
    atexit.register(lambda: [h.close() for h in logger.handlers])

    return logger
