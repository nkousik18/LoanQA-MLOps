"""
run_textract.py
---------------
Stage 2: Runs AWS Textract OCR on PDFs from S3 and saves raw JSON results.

Features:
- PII Masking for privacy protection
- GCS-aware storage (writes to GCS when USE_GCS_OUTPUT=True)
- Skips PDFs already processed (incremental run)
- Executes multiple Textract jobs in parallel
- Structured logs and manifest tracking
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
# --- ADD THIS BLOCK ---
from dotenv import load_dotenv
# --- DEBUGGING & CONFIGURATION (START) ---
# 1. Load .env
# This goes up 3 levels: scripts/aws_extraction_scripts/run_textract.py -> aws_extraction_scripts -> scripts -> PROJECT_ROOT
env_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# 2. Check if the variable exists
key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
print(f"\n🔍 DEBUG: Credential Path from .env: {key_path}")

# 3. Check if the file actually exists on disk
if key_path:
    # Handle relative paths (e.g., llm-microservice/gcs_key.json)
    if not os.path.isabs(key_path):
        # Combine Project Root + Relative Path
        abs_key_path = (env_path.parent / key_path).resolve()
    else:
        abs_key_path = Path(key_path)

    if abs_key_path.exists():
        print(f"✅ DEBUG: Key file FOUND at: {abs_key_path}\n")
        # Force the absolute path back into the environment to be safe
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(abs_key_path)
    else:
        print(f"❌ DEBUG: Key file NOT FOUND at: {abs_key_path}")
        print("   -> Please check the filename and location.\n")
else:
    print("❌ DEBUG: GOOGLE_APPLICATION_CREDENTIALS variable is NOT set. Check your .env file.\n")
# --- DEBUGGING & CONFIGURATION (END) ---
# ----------------------
import boto3
import botocore
# from scripts.aws_extraction_scripts.gcs_utils import upload_to_gcs
from google.cloud import storage
# ---------------------------------------------------------------------
# Ensure project root
# ---------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------
from scripts.aws_extraction_scripts.config import (
    BUCKET,
    REGION,
    RAW_DIR,
    RAW_TEXT_DIR,
    ensure_directories,
)
from scripts.aws_extraction_scripts.gcs_utils import (
    write_json,
    write_text,
    logical_exists,
    read_json,
)
from scripts.aws_extraction_scripts.log_utils import get_logger
from scripts.aws_extraction_scripts.tracker import track_task
from scripts.aws_extraction_scripts.fetch_files import fetch_files, DOC_PREFIX

# PII Masking
from scripts.aws_extraction_scripts.pii_masking import PIIMasker
from scripts.aws_extraction_scripts.monitor_utils import monitor_task
# --- NEW IMPORTS ---
import mlflow
import sentry_sdk # For Alerts
from datetime import datetime

# Initialize Sentry (Optional - if you have a DSN)
# sentry_sdk.init(dsn="your_sentry_dsn")

# Setup MLFlow (Connect to your existing tracking folder)
TRACKING_URI = os.path.join(PROJECT_ROOT, "llm-microservice", "tracking", "mlruns")
mlflow.set_tracking_uri(f"file://{TRACKING_URI}")
mlflow.set_experiment("Textract_OCR_Pipeline")
# ---------------------------------------------------------------------
# Initialize
# ---------------------------------------------------------------------
logger = get_logger("run_textract")
s3 = boto3.client("s3", region_name=REGION)
textract = boto3.client("textract", region_name=REGION)
pii_masker = PIIMasker()  # PII Masker


# ---------------------------------------------------------------------
# Helper: convert Textract blocks to plain text
# ---------------------------------------------------------------------
def textract_blocks_to_text(blocks: List[Dict[str, Any]]) -> str:
    """Convert Textract 'Blocks' into multiline string grouped by page."""
    lines_by_page = {}

    for block in blocks:
        if block.get("BlockType") == "LINE":
            page = block.get("Page", 1)
            text = block.get("Text", "")
            if not text:
                continue
            lines_by_page.setdefault(page, []).append(text)

    pages = []
    for page in sorted(lines_by_page.keys()):
        header = f"=== PAGE {page} ==="
        body = "\n".join(lines_by_page[page])
        pages.append(f"{header}\n{body}")

    return "\n\n".join(pages)


# ---------------------------------------------------------------------
# Safe retry for API calls
# ---------------------------------------------------------------------
# def safe_textract_call(func, **kwargs):
#     """Retries transient network errors during Textract API calls."""
#     for attempt in range(5):
#         try:
#             return func(**kwargs)
#         except (botocore.exceptions.EndpointConnectionError, botocore.exceptions.SSLError) as e:
#             logger.warning(f"Network issue, retrying in 5s: {e}")
#         except Exception as e:
#             logger.warning(f"Unexpected error, retrying in 5s: {e}")
#         time.sleep(5)
#     raise Exception("Failed to connect to Textract after multiple attempts.")

def safe_textract_call(func, **kwargs):
    """Retries with Alerting on final failure."""
    for attempt in range(5):
        try:
            return func(**kwargs)
        except (botocore.exceptions.EndpointConnectionError, botocore.exceptions.SSLError) as e:
            logger.warning(f"Network issue (Attempt {attempt + 1}), retrying: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error (Attempt {attempt + 1}), retrying: {e}")
        time.sleep(5)

    # --- FIX: ALERTING ---
    error_msg = f"CRITICAL: Textract call failed after 5 attempts. Func: {func.__name__}"
    logger.error(error_msg)
    sentry_sdk.capture_message(error_msg)  # Send Alert
    raise Exception(error_msg)
# ---------------------------------------------------------------------
# Fetch all Textract results
# ---------------------------------------------------------------------
def get_all_textract_results(job_id: str) -> Dict[str, Any]:
    """Fetches all Textract pages for a job until complete."""
    pages: List[Dict[str, Any]] = []
    next_token = None
    page_count = 0

    while True:
        if next_token:
            response = safe_textract_call(
                textract.get_document_text_detection,
                JobId=job_id,
                NextToken=next_token,
            )
        else:
            response = safe_textract_call(
                textract.get_document_text_detection,
                JobId=job_id,
            )

        blocks = response.get("Blocks", [])
        pages.extend(blocks)
        page_count += 1
        logger.info(f"Retrieved page {page_count} ({len(blocks)} blocks)")

        next_token = response.get("NextToken")
        if not next_token:
            break
        time.sleep(1)

    logger.info(f"Total blocks retrieved: {len(pages)} across {page_count} pages")
    return {"Blocks": pages}


# ---------------------------------------------------------------------
# Save helpers with PII masking + GCS support
# ---------------------------------------------------------------------
def _save_textract_outputs(pdf_stem: str, full_result: Dict[str, Any]) -> Tuple[str, str]:
    """
    Save Textract results with PII masking to GCS/local.
    
    Steps:
      1. Mask PII in blocks
      2. Save masked JSON (GCS/local via gcs_utils)
      3. Generate and save masked TXT (GCS/local via gcs_utils)
    
    Returns:
        (json_path_str, txt_path_str)
    """
    blocks = full_result.get("Blocks", [])
    
    ### PII MASKING ###
    logger.info(f"🔒 Applying PII masking to {pdf_stem}")
    blocks = pii_masker.mask_textract_blocks(blocks)
    full_result["Blocks"] = blocks  # Update with masked blocks
    ### END PII ###
    
    json_path = RAW_DIR / f"{pdf_stem}_raw.json"
    txt_path = RAW_TEXT_DIR / f"{pdf_stem}_raw.txt"

    # Save JSON (GCS/local)
    write_json(json_path, full_result)

    # Save TXT (GCS/local)
    text_content = textract_blocks_to_text(blocks)
    write_text(txt_path, text_content)

    logger.info(f"✅ Saved masked outputs: {json_path.name} (PII protected)")
    return str(json_path), str(txt_path)


# ---------------------------------------------------------------------
# Single PDF Textract (for sessions)
# ---------------------------------------------------------------------
def run_textract_for_pdf_to_dirs(pdf_key, raw_dir, raw_text_dir):
    """
    Runs Textract OCR for one PDF with PII masking.
    Writes to GCS or local based on USE_GCS_OUTPUT.
    """
    if isinstance(pdf_key, Path):
        s3_key = str(pdf_key).replace("\\", "/")
    else:
        s3_key = str(pdf_key).replace("\\", "/")
        pdf_key = Path(s3_key)

    task_name = f"run_textract_{pdf_key.stem}"
    track_task(task_name, "STARTED")

    try:
        logger.info(f"Starting Textract for S3 key: {s3_key}")
        start_response = safe_textract_call(
            textract.start_document_text_detection,
            DocumentLocation={"S3Object": {"Bucket": BUCKET, "Name": s3_key}},
        )
        job_id = start_response["JobId"]
        logger.info(f"Textract job started: {job_id}")

        # Poll for completion
        while True:
            result = safe_textract_call(
                textract.get_document_text_detection,
                JobId=job_id,
            )
            status = result["JobStatus"]
            if status in ["SUCCEEDED", "FAILED"]:
                break
            logger.info(f"Waiting for {s3_key} (status={status})")
            time.sleep(5)

        if status != "SUCCEEDED":
            msg = f"Textract failed for {s3_key}"
            logger.error(msg)
            track_task(task_name, "FAILED", error=msg)
            return None, None

        full_result = get_all_textract_results(job_id)
        blocks = full_result.get("Blocks", [])

        ### PII MASKING ###
        logger.info(f"🔒 Applying PII masking")
        blocks = pii_masker.mask_textract_blocks(blocks)
        full_result["Blocks"] = blocks
        ### END PII ###

        pdf_stem = pdf_key.stem
        json_path = Path(raw_dir) / f"{pdf_stem}_raw.json"
        txt_path = Path(raw_text_dir) / f"{pdf_stem}_raw.txt"

        # Write using GCS utils
        write_json(json_path, full_result)
        text_content = textract_blocks_to_text(blocks)
        write_text(txt_path, text_content)

        msg = f"Saved masked outputs: {json_path} and {txt_path}"
        logger.info(msg)
        track_task(task_name, "SUCCESS", details=msg)
        return str(json_path), str(txt_path)

    except Exception as e:
        err = f"Error running Textract for {s3_key}: {e}"
        logger.exception(err)
        track_task(task_name, "FAILED", error=str(e))
        return None, None


# ---------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------
def run_textract_for_pdf(pdf_key):
    """Backwards-compatible wrapper for batch pipeline."""
    json_path, _ = run_textract_for_pdf_to_dirs(pdf_key, RAW_DIR, RAW_TEXT_DIR)
    return json_path


# ---------------------------------------------------------------------
# Batch Textract with PII + GCS
# ---------------------------------------------------------------------
# def run_textract_all(**context):
#     """
#     Runs Textract OCR on all PDFs with PII masking and GCS storage.
#     """
#     task_name = "run_textract_all"
#     track_task(task_name, "STARTED")
#     ensure_directories()
#
#     pdfs = fetch_files(prefix=DOC_PREFIX)
#     if not pdfs:
#         msg = "No PDFs found."
#         logger.warning(msg)
#         track_task(task_name, "SUCCESS", details=msg)
#         return []
#
#     jobs: List[Dict[str, Any]] = []
#     output_files: List[str] = []
#
#     # Check what needs processing
#     for pdf_key in pdfs:
#         pdf_stem = Path(pdf_key).stem
#         json_path = RAW_DIR / f"{pdf_stem}_raw.json"
#         txt_path = RAW_TEXT_DIR / f"{pdf_stem}_raw.txt"
#
#         json_exists = logical_exists(json_path)
#         txt_exists = logical_exists(txt_path)
#
#         # Case 1: both exist -> skip
#         if json_exists and txt_exists:
#             logger.info(f"Skipping {pdf_key} (already processed)")
#             output_files.append(str(json_path))
#             continue
#
#         # Case 2: JSON exists, TXT missing -> regenerate TXT
#         if json_exists and not txt_exists:
#             logger.info(f"Regenerating TXT for {pdf_key}")
#             try:
#                 data = read_json(json_path)
#                 blocks = data.get("Blocks", [])
#
#                 # Re-mask in case old JSON wasn't masked
#                 blocks = pii_masker.mask_textract_blocks(blocks)
#
#                 text_content = textract_blocks_to_text(blocks)
#                 write_text(txt_path, text_content)
#                 output_files.append(str(json_path))
#             except Exception as e:
#                 logger.exception(f"Failed to regenerate TXT: {e}")
#             continue
#
#         # Case 3: Need to run Textract
#         per_file_task = f"run_textract_{pdf_stem}"
#         track_task(per_file_task, "STARTED")
#
#         try:
#             start_response = safe_textract_call(
#                 textract.start_document_text_detection,
#                 DocumentLocation={"S3Object": {"Bucket": BUCKET, "Name": pdf_key}},
#             )
#             job_id = start_response["JobId"]
#             logger.info(f"Started Textract: {pdf_key} -> {job_id}")
#             jobs.append({
#                 "pdf_key": pdf_key,
#                 "pdf_stem": pdf_stem,
#                 "job_id": job_id,
#                 "task_name": per_file_task,
#             })
#         except Exception as e:
#             logger.exception(f"Error starting Textract: {e}")
#             track_task(per_file_task, "FAILED", error=str(e))
#
#     if not jobs:
#         msg = "No new Textract jobs needed"
#         logger.info(msg)
#         track_task(task_name, "SUCCESS", details=msg)
#         return output_files
#
#     # Poll until all jobs complete
#     POLL_INTERVAL = 5
#     while jobs:
#         for job in list(jobs):
#             try:
#                 result = safe_textract_call(
#                     textract.get_document_text_detection,
#                     JobId=job["job_id"],
#                 )
#                 status = result.get("JobStatus")
#             except Exception as e:
#                 logger.exception(f"Error checking status: {e}")
#                 track_task(job["task_name"], "FAILED", error=str(e))
#                 jobs.remove(job)
#                 continue
#
#             if status == "SUCCEEDED":
#                 full_result = get_all_textract_results(job["job_id"])
#
#                 # Save with PII masking + GCS
#                 json_path_str, txt_path_str = _save_textract_outputs(
#                     job["pdf_stem"], full_result
#                 )
#
#                 logger.info(f"✅ Saved: {json_path_str}")
#                 track_task(job["task_name"], "SUCCESS")
#                 output_files.append(json_path_str)
#                 jobs.remove(job)
#
#             elif status == "FAILED":
#                 logger.error(f"❌ Textract failed: {job['pdf_key']}")
#                 track_task(job["task_name"], "FAILED")
#                 jobs.remove(job)
#             else:
#                 logger.info(f"⏳ Waiting: {job['pdf_key']} ({status})")
#
#         time.sleep(POLL_INTERVAL)
#
#     msg = f"✅ Completed Textract for {len(output_files)} files"
#     logger.info(msg)
#     track_task(task_name, "SUCCESS", details=msg)
#     return output_files
@monitor_task("Batch_Textract_Processing")
def run_textract_all(**context):
    task_name = "run_textract_all"
    track_task(task_name, "STARTED")
    ensure_directories()

    # Optional: Set a tag on the ACTIVE run started by the decorator
    mlflow.set_tag("mlflow.runName", f"Batch_Textract_{int(time.time())}")

    pdfs = fetch_files(prefix=DOC_PREFIX)
    mlflow.log_param("total_pdfs_found", len(pdfs))

    if not pdfs:
        msg = "No PDFs found."
        logger.warning(msg)
        return []

    jobs = []
    output_files = []

    # --- 1. SUBMISSION LOOP ---
    for pdf_key in pdfs:
        pdf_stem = Path(pdf_key).stem

        # Check if output already exists (Skip logic)
        json_path = RAW_DIR / f"{pdf_stem}_raw.json"
        txt_path = RAW_TEXT_DIR / f"{pdf_stem}_raw.txt"
        if logical_exists(json_path) and logical_exists(txt_path):
            logger.info(f"Skipping {pdf_key} (already processed)")
            output_files.append(str(json_path))
            continue

        # Start Job with MLFlow Logging
        try:
            start_response = safe_textract_call(
                textract.start_document_text_detection,
                DocumentLocation={"S3Object": {"Bucket": BUCKET, "Name": pdf_key}}
            )

            job_id = start_response["JobId"]
            logger.info(f"Started Textract: {pdf_key} -> {job_id}")

            jobs.append({
                "pdf_key": pdf_key,
                "pdf_stem": pdf_stem,
                "job_id": job_id,
                "start_time": time.time(),
                "task_name": f"OCR_{pdf_stem}"
            })
        except Exception as e:
            logger.error(f"Failed to submit {pdf_key}: {e}")
            # Log failure to MLFlow
            with mlflow.start_run(run_name=f"Fail_{pdf_stem}", nested=True):
                mlflow.log_param("status", "SUBMISSION_FAILED")
                mlflow.log_param("error", str(e))
            continue

    if not jobs:
        logger.info("No new jobs to process.")
        return output_files

    # --- 2. POLLING LOOP ---
    POLL_INTERVAL = 5
    while jobs:
        for job in list(jobs):
            try:
                result = safe_textract_call(
                    textract.get_document_text_detection,
                    JobId=job["job_id"],
                )
                status = result.get("JobStatus")
            except Exception as e:
                logger.exception(f"Error checking status for {job['pdf_key']}: {e}")
                jobs.remove(job)
                continue

            if status == "SUCCEEDED":
                # --- MLFLOW: Track Successful File ---
                with mlflow.start_run(run_name=f"OCR_{job['pdf_stem']}", nested=True):
                    duration = time.time() - job['start_time']

                    full_result = get_all_textract_results(job["job_id"])
                    page_count = len(full_result.get("Blocks", []))

                    # Save Data (PII Masking + GCS Upload)
                    json_path_str, txt_path_str = _save_textract_outputs(
                        job["pdf_stem"], full_result
                    )

                    # Log Metrics
                    mlflow.log_metric("duration_seconds", duration)
                    mlflow.log_metric("page_count", page_count)
                    mlflow.log_param("output_path", json_path_str)

                    logger.info(f"✅ Completed: {job['pdf_key']}")
                    output_files.append(json_path_str)

                jobs.remove(job)

            elif status == "FAILED":
                # --- MLFLOW: Track Failure ---
                with mlflow.start_run(run_name=f"Fail_{job['pdf_stem']}", nested=True):
                    mlflow.log_param("status", "TEXTRACT_JOB_FAILED")
                    logger.error(f"❌ Textract job failed for {job['pdf_key']}")
                jobs.remove(job)
            else:
                # Still IN_PROGRESS, wait
                pass

        time.sleep(POLL_INTERVAL)

    msg = f"✅ Completed Textract for {len(output_files)} files"
    logger.info(msg)
    track_task(task_name, "SUCCESS", details=msg)
    return output_files
if __name__ == "__main__":
    ensure_directories()
    run_textract_all()
