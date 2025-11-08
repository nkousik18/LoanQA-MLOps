"""
config.py
----------
Central configuration for the AWS Extraction OCR → Textract → Normalization pipeline.

Features:
- Modular and reusable: all scripts import paths dynamically.
- Works in both VS Code (local) and Airflow (Docker) environments.
- Single source of truth for all directory references.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------
# 🌐 AWS SETTINGS
# ---------------------------------------------------------------------
BUCKET = os.getenv("S3_BUCKET", "textract-bucket-yash07")
REGION = os.getenv("AWS_REGION", "us-east-1")

# ---------------------------------------------------------------------
# 🧭 PROJECT ROOT
# ---------------------------------------------------------------------
# Resolve project root dynamically (two levels above this config file)
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../doc-understand

# ---------------------------------------------------------------------
# 📂 BASE DIRECTORIES
# ---------------------------------------------------------------------
DATA_ROOT = PROJECT_ROOT / "data" / "aws_extraction_data"
LOGS_ROOT = PROJECT_ROOT / "logs" / "aws_extraction_logs"
REPORTS_ROOT = PROJECT_ROOT / "reports" / "aws_extraction_reports"

# ---------------------------------------------------------------------
# 🧱 DATA SUBDIRECTORIES
# ---------------------------------------------------------------------
RAW_DIR = DATA_ROOT / "raw"
SEGMENTED_DIR = DATA_ROOT / "segmented"
NORMALIZED_DIR = DATA_ROOT / "normalized"
SCHEMA_DIR = DATA_ROOT / "schema"

# ---------------------------------------------------------------------
# 🧾 LOG DIRECTORY (auto-switch for Airflow)
# ---------------------------------------------------------------------
AIRFLOW_HOME = Path("/opt/airflow")

if AIRFLOW_HOME.exists():
    LOG_DIR = AIRFLOW_HOME / "logs" / "aws_extraction_logs"
else:
    LOG_DIR = LOGS_ROOT

# ---------------------------------------------------------------------
# ⚙️ RUNTIME SETTINGS
# ---------------------------------------------------------------------
MAX_RETRIES = 5
RETRY_DELAY = 5  # seconds

# ---------------------------------------------------------------------
# 🧩 Helper Function
# ---------------------------------------------------------------------
def ensure_directories() -> None:
    """Create all required directories if they don’t exist."""
    for d in [RAW_DIR, SEGMENTED_DIR, NORMALIZED_DIR, SCHEMA_DIR, LOG_DIR, REPORTS_ROOT]:
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# 🧾 Path Verification (print all paths clearly)
# ---------------------------------------------------------------------
def print_all_paths():
    print("\n📁 ==== CONFIGURATION PATHS ====")
    print(f"Project Root       : {PROJECT_ROOT}")
    print(f"Data Root          : {DATA_ROOT}")
    print(f" ├── Raw Data      : {RAW_DIR}")
    print(f" ├── Segmented     : {SEGMENTED_DIR}")
    print(f" ├── Normalized    : {NORMALIZED_DIR}")
    print(f" └── Schema        : {SCHEMA_DIR}")
    print(f"Logs Root          : {LOGS_ROOT}")
    print(f"Reports Root       : {REPORTS_ROOT}")
    print(f"Active Log Dir     : {LOG_DIR}")
    print(f"AWS Bucket         : {BUCKET}")
    print(f"AWS Region         : {REGION}")
    print("====================================\n")


if __name__ == "__main__":
    ensure_directories()
    print_all_paths()
    print("✅ Directory structure verified and paths printed successfully.")
