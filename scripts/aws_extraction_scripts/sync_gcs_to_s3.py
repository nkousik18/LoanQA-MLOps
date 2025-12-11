"""
sync_gcs_to_s3.py
-----------------
Step 0: Sync PDFs from a GCP Storage bucket into our AWS S3 bucket.

We KEEP the existing AWS/Textract pipeline exactly as is:
- S3 bucket: BUCKET from config.py (e.g., textract-bucket-yash07)
- Prefixes in S3:
      docs/          (batch corpus)
      user_uploads/  (single-PDF user flow)

GCS side:
- Uses the same bucket + logical paths as config.py:
      GCS_BUCKET      (from config.py)
      DOCS_DIR        -> to_gcs_key(DOCS_DIR)  (e.g. "data/docs")
      USER_UPLOADS_DIR -> to_gcs_key(USER_UPLOADS_DIR)
"""

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------
# Project root + imports (same style as your other aws_extraction scripts)
# ---------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent      # scripts/aws_extraction_scripts
PROJECT_ROOT = CURRENT_DIR.parents[1]              # doc-understand
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from google.cloud import storage
import boto3

from scripts.aws_extraction_scripts.config import (
    BUCKET,
    REGION,
    GCS_BUCKET,
    DOCS_DIR,
    USER_UPLOADS_DIR,
    to_gcs_key,
)
from scripts.aws_extraction_scripts.log_utils import get_logger
from scripts.aws_extraction_scripts.tracker import track_task

LOGGER = get_logger(__name__)


# ---------------------------------------------------------------------
# Core sync helper
# ---------------------------------------------------------------------
def _sync_prefix(gcs_prefix: str, s3_prefix: str) -> int:
    """
    Copy all .pdf files from GCS `gcs_prefix` into S3 `s3_prefix`,
    preserving filenames relative to the prefix.

    Example:
      gcs_prefix = "data/docs/"
      s3_prefix  = "docs/"
      GCS object "data/docs/loan1.pdf" -> S3 key "docs/loan1.pdf"

    Returns:
      Number of files successfully copied.
    """
    if not gcs_prefix.endswith("/"):
        gcs_prefix = gcs_prefix + "/"

    LOGGER.info(
        f"Starting GCS→S3 sync: "
        f"bucket={GCS_BUCKET}, gcs_prefix='{gcs_prefix}', s3_prefix='{s3_prefix}'"
    )

    gcs_client = storage.Client()
    s3_client = boto3.client("s3", region_name=REGION)

    blobs = gcs_client.list_blobs(GCS_BUCKET, prefix=gcs_prefix)

    count = 0
    for blob in blobs:
        name = blob.name

        # Skip folders and non-PDFs
        if not name.lower().endswith(".pdf"):
            continue

        # Make name relative to the prefix
        relative_name = name[len(gcs_prefix):]
        if not relative_name or relative_name.endswith("/"):
            continue

        s3_key = f"{s3_prefix}{relative_name}"

        LOGGER.info(f"Syncing gs://{GCS_BUCKET}/{name} -> s3://{BUCKET}/{s3_key}")

        data = blob.download_as_bytes()

        s3_client.put_object(
            Bucket=BUCKET,
            Key=s3_key,
            Body=data,
            ContentType="application/pdf",
        )
        count += 1

    LOGGER.info(
        f"Finished GCS→S3 sync for prefix '{gcs_prefix}'. "
        f"Files copied: {count}"
    )
    return count


# ---------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------
def sync_docs() -> int:
    """Sync only batch corpus PDFs: GCS DOCS_DIR -> S3 docs/."""
    task_name = "sync_docs"
    track_task(task_name, "STARTED")

    try:
        gcs_docs_prefix = to_gcs_key(DOCS_DIR)
        copied = _sync_prefix(gcs_docs_prefix, "docs/")
        msg = f"sync_docs completed; files copied: {copied}"
        LOGGER.info(msg)
        track_task(task_name, "SUCCESS", details=msg)
        return copied
    except Exception as e:
        err = f"sync_docs failed: {e}"
        LOGGER.exception(err)
        track_task(task_name, "FAILED", error=str(e))
        return 0


def sync_user_uploads() -> int:
    """Sync only single-PDF user uploads: GCS USER_UPLOADS_DIR -> S3 user_uploads/."""
    task_name = "sync_user_uploads"
    track_task(task_name, "STARTED")

    try:
        gcs_upload_prefix = to_gcs_key(USER_UPLOADS_DIR)
        copied = _sync_prefix(gcs_upload_prefix, "user_uploads/")
        msg = f"sync_user_uploads completed; files copied: {copied}"
        LOGGER.info(msg)
        track_task(task_name, "SUCCESS", details=msg)
        return copied
    except Exception as e:
        err = f"sync_user_uploads failed: {e}"
        LOGGER.exception(err)
        track_task(task_name, "FAILED", error=str(e))
        return 0


def sync_docs_and_uploads() -> int:
    """Sync both batch docs and user uploads."""
    task_name = "sync_docs_and_uploads"
    track_task(task_name, "STARTED")

    try:
        copied_docs = sync_docs()
        copied_uploads = sync_user_uploads()
        total = copied_docs + copied_uploads
        msg = (
            f"sync_docs_and_uploads completed; "
            f"docs={copied_docs}, uploads={copied_uploads}, total={total}"
        )
        LOGGER.info(msg)
        track_task(task_name, "SUCCESS", details=msg)
        return total
    except Exception as e:
        err = f"sync_docs_and_uploads failed: {e}"
        LOGGER.exception(err)
        track_task(task_name, "FAILED", error=str(e))
        return 0


# ---------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # By default, sync both. You can change to sync_docs() if you only want batch.
    sync_docs_and_uploads()
