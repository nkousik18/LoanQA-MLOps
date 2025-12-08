"""
tracker.py
-----------
Tracks progress and logs pipeline stage statuses to
reports/aws_extraction_reports/manifest.json

With GCS support:
- manifest.json is addressed as a logical path under REPORTS_ROOT.
- Actual storage location (GCS vs local vs both) is controlled by config:
    USE_GCS_OUTPUT / WRITE_LOCAL_COPY
"""

import os
import sys
import json
import datetime
from pathlib import Path
from typing import Any, List

# ---------------------------------------------------------------------
# 🔧 Ensure project root is on sys.path (works in VS Code + Airflow)
# ---------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# ---------------------------------------------------------------------
# 📦 Imports from centralized config + logger + GCS utils
# ---------------------------------------------------------------------
from scripts.aws_extraction_scripts.config import REPORTS_ROOT
from scripts.aws_extraction_scripts.log_utils import get_logger
from scripts.aws_extraction_scripts.gcs_utils import (
    write_json,
    read_json,
    logical_exists,
)

logger = get_logger("tracker")


# ---------------------------------------------------------------------
# 🔔 Real-time alert hook
# ---------------------------------------------------------------------
def alert_on_error(message: str):
    """Prints an alert when errors or failures are detected."""
    if "ERROR" in message.upper() or "FAILED" in message.upper():
        print(f"🚨 ALERT: {message}")  # Visible in terminal / Airflow logs
        # Future: Add Slack/email/SNS integration here


# ---------------------------------------------------------------------
# 🧩 Main tracking function (GCS-aware)
# ---------------------------------------------------------------------
def track_task(task_name: str, status: str, details: str | None = None, error: str | None = None):
    """
    Records task status for auditing and monitoring.

    Writes entries to:
        REPORTS_ROOT / "manifest.json"
    using GCS-aware read/write helpers.

    - If manifest.json exists (in GCS or local), append to the list.
    - If it doesn't exist, create a new list.
    """
    manifest_path: Path = REPORTS_ROOT / "manifest.json"

    entry = {
        "task": task_name,
        "status": status.upper(),
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "details": details or "",
        "error": error or "",
    }

    try:
        # Load existing manifest safely (from GCS or local)
        existing: List[Any] = []
        if logical_exists(manifest_path):
            try:
                existing_raw = read_json(manifest_path)
                if isinstance(existing_raw, list):
                    existing = existing_raw
                elif isinstance(existing_raw, dict):
                    # In case an old version stored a single dict
                    existing = [existing_raw]
                else:
                    existing = []
            except Exception as e:
                logger.warning(f"⚠️ Could not read existing manifest, resetting: {e}")
                existing = []

        # Append new entry and write back via gcs_utils
        existing.append(entry)
        write_json(manifest_path, existing)

        # Log success
        msg = f"🧾 Recorded task '{task_name}' as {status}"
        logger.info(msg)

        # Handle errors / failures
        if error or status.upper() == "FAILED":
            err_msg = f"{task_name} - {error or 'Unknown issue'}"
            logger.error(f"❌ {err_msg}")
            alert_on_error(err_msg)

    except Exception as e:
        logger.exception(f"❌ Failed to update manifest for {task_name}: {e}")


# ---------------------------------------------------------------------
# 🧠 Example run (for manual testing)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    track_task("fetch_files", "STARTED")
    track_task("fetch_files", "SUCCESS", details="Fetched 10 files from S3.")
    track_task("normalize_text", "FAILED", error="Empty JSON structure detected.")
