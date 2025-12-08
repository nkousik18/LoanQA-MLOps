"""
fetch_files.py
--------------
Stage 1: Fetches PDF files for the AWS Textract pipeline.

GCS-aware behavior:
- When USE_GCS_OUTPUT=True:
    * Source-of-truth PDFs live in GCS under data/docs/ (DOCS_DIR).
    * This script:
        1) Lists PDFs from GCS.
        2) Ensures each is present in S3 under "docs/<filename>.pdf"
           (uploads if missing).
        3) Returns the corresponding S3 keys for downstream Textract.

- When USE_GCS_OUTPUT=False:
    * Falls back to listing PDFs directly from S3 under DOC_PREFIX.
"""

import os, sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from google.cloud import storage

# ---------------------------------------------------------------------
# 🔧 Ensure working directory and imports
# ---------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.aws_extraction_scripts.config import (
    BUCKET,
    REGION,
    GCS_BUCKET,
    USE_GCS_OUTPUT,
    DOCS_DIR,
    to_gcs_key,
)
from scripts.aws_extraction_scripts.log_utils import get_logger
from scripts.aws_extraction_scripts.tracker import track_task

# ---------------------------------------------------------------------
# 📁 S3 prefixes
# ---------------------------------------------------------------------
DOC_PREFIX = "docs/"          # batch / corpus docs in S3
UPLOAD_PREFIX = "user_uploads/"  # interactive user uploads (single-PDF flow only)


# ---------------------------------------------------------------------
# 🧩 Internal helpers
# ---------------------------------------------------------------------
def _list_pdfs_from_s3(prefix: str, file_extension: str, logger):
    """Old behavior: list PDFs directly from S3."""
    s3 = boto3.client("s3", region_name=REGION)
    logger.info(f"🚀 [S3] Fetching files from bucket: {BUCKET}, prefix: '{prefix}'")

    response = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
    pdf_keys = [
        obj["Key"]
        for obj in response.get("Contents", [])
        if obj["Key"].lower().endswith(file_extension.lower())
        and not obj["Key"].startswith(UPLOAD_PREFIX)  # keep hard guard
    ]
    return pdf_keys


def _sync_pdfs_from_gcs_to_s3(file_extension: str, logger):
    """
    New behavior when USE_GCS_OUTPUT=True.

    - Lists PDFs from GCS under DOCS_DIR (logical 'data/docs/').
    - Ensures each PDF is present in S3 at 'docs/<filename>.pdf'.
    - Returns the list of S3 keys for those PDFs.
    """
    task_name = "fetch_files"
    track_task(task_name, "STARTED")

    storage_client = storage.Client()
    s3 = boto3.client("s3", region_name=REGION)

    docs_prefix = to_gcs_key(DOCS_DIR)  # e.g. "data/docs"
    if not docs_prefix.endswith("/"):
        docs_prefix = docs_prefix + "/"

    logger.info(
        f"🚀 [GCS] Listing PDFs in bucket '{GCS_BUCKET}' under prefix '{docs_prefix}'"
    )

    blobs = storage_client.list_blobs(GCS_BUCKET, prefix=docs_prefix)
    gcs_pdf_blobs = [
        blob for blob in blobs if blob.name.lower().endswith(file_extension.lower())
    ]

    if not gcs_pdf_blobs:
        msg = "⚠️ No matching PDF files found in GCS under data/docs/."
        logger.warning(msg)
        track_task(task_name, "SUCCESS", details=msg)
        return []

    s3_keys = []

    for blob in gcs_pdf_blobs:
        # blob.name looks like "data/docs/loan1.pdf"
        rel_path = blob.name[len(docs_prefix) :]  # "loan1.pdf" or "subdir/loan1.pdf"
        if not rel_path or rel_path.endswith("/"):
            # skip any 'directory' markers
            continue

        s3_key = DOC_PREFIX + rel_path.replace("\\", "/")  # "docs/loan1.pdf"
        logger.info(f"🔁 Ensuring S3 has: s3://{BUCKET}/{s3_key}")

        # Check if already in S3
        try:
            s3.head_object(Bucket=BUCKET, Key=s3_key)
            logger.info(f"✅ Already present in S3: {s3_key}")
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in ("404", "NoSuchKey", "NotFound"):
                logger.info(f"⬆️ Uploading from GCS → S3: {blob.name} → {s3_key}")
                data = blob.download_as_bytes()
                s3.put_object(Bucket=BUCKET, Key=s3_key, Body=data)
                logger.info(f"✅ Uploaded to S3: {s3_key}")
            else:
                logger.exception(
                    f"❌ Unexpected error checking S3 object {s3_key}: {e}"
                )
                # We don't fail the whole batch, just skip this one.
                continue

        s3_keys.append(s3_key)

    if not s3_keys:
        msg = "⚠️ No valid PDFs synced from GCS to S3."
        logger.warning(msg)
        track_task(task_name, "SUCCESS", details=msg)
    else:
        msg = f"✅ Synced {len(s3_keys)} PDFs from GCS to S3."
        logger.info(msg)
        track_task(task_name, "SUCCESS", details=msg)

    logger.info(f"📦 Returning S3 keys for Textract: {s3_keys}")
    return s3_keys


# ---------------------------------------------------------------------
# 🚀 Fetch function (public)
# ---------------------------------------------------------------------
def fetch_files(prefix: str = DOC_PREFIX, file_extension: str = ".pdf", **context):
    """
    Lists PDF files for batch processing.

    When USE_GCS_OUTPUT=True:
        - Ignores the S3 prefix parameter and instead:
            * reads PDFs from GCS under data/docs/
            * syncs them to S3 at docs/<filename>.pdf
            * returns those S3 keys

    When USE_GCS_OUTPUT=False:
        - Uses the original behavior:
            * lists PDFs in S3 under `prefix`
            * skips any keys under UPLOAD_PREFIX
    """
    logger = get_logger("fetch_files")

    # GCS-driven source of truth
    if USE_GCS_OUTPUT:
        return _sync_pdfs_from_gcs_to_s3(file_extension=file_extension, logger=logger)

    # Legacy S3-only mode
    task_name = "fetch_files"
    track_task(task_name, "STARTED")

    try:
        pdf_keys = _list_pdfs_from_s3(prefix, file_extension, logger)

        if not pdf_keys:
            msg = "⚠️ No matching files found in S3 bucket."
            logger.warning(msg)
            track_task(task_name, "SUCCESS", details=msg)
        else:
            msg = f"✅ Found {len(pdf_keys)} files in S3: {pdf_keys}"
            logger.info(msg)
            track_task(task_name, "SUCCESS", details=msg)

        logger.info(f"📦 Returning fetched S3 keys: {pdf_keys}")
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
