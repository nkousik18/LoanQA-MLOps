import os
import logging
from logging import Logger
from datetime import datetime

# ============================================================
# 1️⃣ Project Paths and Constants (Auto-detect Project Root)
# ============================================================

def find_project_root(start_path: str) -> str:
    """Traverse upward to find the project root (contains /data or /.git or /.venv)."""
    current = os.path.abspath(start_path)
    while current != os.path.dirname(current):
        if any(os.path.isdir(os.path.join(current, marker)) for marker in ["data", ".git", ".venv"]):
            return current
        current = os.path.dirname(current)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# --- Auto-resolve project root (e.g., DL_project/) ---
BASE_DIR = find_project_root(os.path.dirname(__file__))

# === Central Data Directories ===
DATA_DIR = os.path.join(BASE_DIR, "data", "loan_docs")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "clean_texts")

# === Centralized Log Directories ===
LOG_DIR = os.path.join(BASE_DIR, "logs")
EXTRACTION_LOG_DIR = os.path.join(LOG_DIR, "extraction_logs")
LLM_LOG_DIR = os.path.join(LOG_DIR, "llm_logs")
DAG_LOG_DIR = os.path.join(LOG_DIR, "dag_logs")
TEST_LOG_DIR = os.path.join(LOG_DIR, "test_logs")

# 🆕 New anomaly-specific log folder
ANOMALY_LOG_DIR = os.path.join(LOG_DIR, "anomaly_logs")

# Create all log subfolders
for d in [LOG_DIR, EXTRACTION_LOG_DIR, LLM_LOG_DIR, DAG_LOG_DIR, TEST_LOG_DIR, ANOMALY_LOG_DIR]:  # 🆕
    os.makedirs(d, exist_ok=True)

# Global constants
MIN_TEXT_LEN = 40
LANG = "en"

# Console summary
print(f"[CONFIG] Base Directory   : {BASE_DIR}")
print(f"[CONFIG] Data Directory   : {DATA_DIR}")
print(f"[CONFIG] Output Directory : {OUTPUT_DIR}")
print(f"[CONFIG] Log Directory    : {LOG_DIR}")

# ============================================================
# 2️⃣ Centralized Logger Factory
# ============================================================

def setup_logger(name: str, log_type: str = "extraction") -> Logger:
    """
    Initializes and returns a centralized logger.
    log_type options → 'extraction', 'llm', 'dag', 'test', 'preprocessing', 'anomaly'
    Each type goes into its own subdirectory under /logs/.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Map log type to its directory
    log_subdir_map = {
        "extraction": EXTRACTION_LOG_DIR,
        "llm": LLM_LOG_DIR,
        "dag": DAG_LOG_DIR,
        "test": TEST_LOG_DIR,
        "preprocessing": EXTRACTION_LOG_DIR,  # ✅ send preprocessing logs to extraction_logs folder
        "anomaly": ANOMALY_LOG_DIR,           # 🆕 dedicated anomaly logging
    }

    # Pick the correct folder; fallback to base /logs/ if undefined
    target_dir = log_subdir_map.get(log_type.lower(), LOG_DIR)
    os.makedirs(target_dir, exist_ok=True)

    # 🆕 File naming convention for anomaly logs
    if log_type.lower() == "anomaly":
        log_file = os.path.join(target_dir, f"anomaly_{timestamp}.log")
    else:
        log_file = os.path.join(target_dir, f"{log_type}_{timestamp}.log")

    # Configure logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers if reimported
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    logger.info(f"🧩 Logger initialized for {log_type.upper()} → {log_file}")
    return logger
