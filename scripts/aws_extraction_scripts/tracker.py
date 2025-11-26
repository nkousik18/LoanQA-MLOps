"""
tracker.py
-----------
Tracks progress and logs pipeline stage statuses to reports/aws_extraction_reports/manifest.json
with automatic alerting for failed or error states.
"""

import os, sys, json, datetime

# ---------------------------------------------------------------------
# 🔧 Ensure project root is on sys.path (works in VS Code + Airflow)
# ---------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# ---------------------------------------------------------------------
# 📦 Imports from your centralized config + logger
# ---------------------------------------------------------------------
from scripts.aws_extraction_scripts.config import REPORTS_ROOT
from scripts.aws_extraction_scripts.log_utils import get_logger

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
# 🧩 Main tracking function
# ---------------------------------------------------------------------
def track_task(task_name, status, details=None, error=None):
    """
    Records task status for auditing and monitoring.
    Writes entries to reports/aws_extraction_reports/manifest.json and triggers alerts for failures.
    """
    os.makedirs(REPORTS_ROOT, exist_ok=True)
    manifest_path = os.path.join(REPORTS_ROOT, "manifest.json")

    entry = {
        "task": task_name,
        "status": status.upper(),
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "details": details or "",
        "error": error or "",
    }

    try:
        # Load existing manifest safely
        existing = []
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    existing = json.loads(content)

        # Append new entry
        existing.append(entry)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)

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
