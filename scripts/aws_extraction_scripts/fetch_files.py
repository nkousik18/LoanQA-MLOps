"""
fetch_files.py
--------------
Stage 1: Fetches PDF files from AWS S3 and logs them.
Compatible with both Airflow and local (VS Code) runs.
"""

import boto3
from botocore.exceptions import ClientError
import os, sys

# ---------------------------------------------------------------------
# 🔧 Ensure working directory and imports
# ---------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.aws_extraction_scripts.config import BUCKET, REGION
from scripts.aws_extraction_scripts.log_utils import get_logger
from scripts.aws_extraction_scripts.tracker import track_task

# ---------------------------------------------------------------------
# 🚀 Fetch function
# ---------------------------------------------------------------------
def fetch_files(prefix: str = "", file_extension: str = ".pdf", **context):
    """
    Lists all PDF files in the configured S3 bucket.
    Returns a list of matching S3 keys.
    """
    logger = get_logger("fetch_files")   # ✅ initialize logger inside
    s3 = boto3.client("s3", region_name=REGION)
    task_name = "fetch_files"
    track_task(task_name, "STARTED")

    try:
        logger.info(f"🚀 Fetching files from bucket: {BUCKET}, prefix: '{prefix}'")
        response = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)

        pdf_keys = [
            obj["Key"]
            for obj in response.get("Contents", [])
            if obj["Key"].endswith(file_extension)
        ]

        if not pdf_keys:
            msg = "⚠️ No matching files found in bucket."
            logger.warning(msg)
            track_task(task_name, "SUCCESS", details=msg)
        else:
            msg = f"✅ Found {len(pdf_keys)} files: {pdf_keys}"
            logger.info(msg)
            track_task(task_name, "SUCCESS", details=msg)

        logger.info(f"📦 Returning fetched files: {pdf_keys}")
        return pdf_keys

    except ClientError as e:
        logger.exception(f"❌ S3 client error: {e}")
        track_task(task_name, "FAILED", error=str(e))
        return []

    except Exception as e:
        logger.exception(f"❌ Unexpected error while fetching files: {e}")
        track_task(task_name, "FAILED", error=str(e))
        return []

# ---------------------------------------------------------------------
# 🧩 Optional standalone entry point (manual test)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    fetched = fetch_files()
    print(f"📄 Manually fetched {len(fetched)} files → {fetched}")
