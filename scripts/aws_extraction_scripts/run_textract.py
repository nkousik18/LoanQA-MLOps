"""
run_textract.py
---------------
Stage 2: Runs AWS Textract OCR on PDFs from S3 and saves raw JSON results.

Features:
- Skips PDFs already processed (incremental run)
- Executes multiple Textract jobs in parallel (AWS-side parallelism)
- Structured logs and manifest tracking
- Works in both VS Code (local) and Airflow (Docker) environments
"""

import os
import sys
import json
import time
import boto3
import botocore
from pathlib import Path

# ---------------------------------------------------------------------
# 🔧 Ensure project root dynamically (works in both local & Docker)
# ---------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# ---------------------------------------------------------------------
# 📦 Centralized imports
# ---------------------------------------------------------------------
from scripts.aws_extraction_scripts.config import BUCKET, REGION, RAW_DIR
from scripts.aws_extraction_scripts.log_utils import get_logger
from scripts.aws_extraction_scripts.tracker import track_task
from scripts.aws_extraction_scripts.fetch_files import fetch_files

# ---------------------------------------------------------------------
# ⚙️ Initialize clients and logger
# ---------------------------------------------------------------------
logger = get_logger("run_textract")
s3 = boto3.client("s3", region_name=REGION)
textract = boto3.client("textract", region_name=REGION)

# ---------------------------------------------------------------------
# 🔁 Safe retry for API calls
# ---------------------------------------------------------------------
def safe_textract_call(func, **kwargs):
    """Retries transient network errors during Textract API calls."""
    for attempt in range(5):
        try:
            return func(**kwargs)
        except (botocore.exceptions.EndpointConnectionError, botocore.exceptions.SSLError) as e:
            logger.warning(f"⚠️ Network issue, retrying in 5s… {e}")
        except Exception as e:
            logger.warning(f"⚠️ Unexpected error: {e}, retrying in 5s…")
        time.sleep(5)
    raise Exception("❌ Failed to connect after multiple attempts.")

# ---------------------------------------------------------------------
# 📑 Fetch all Textract results for a job
# ---------------------------------------------------------------------
def get_all_textract_results(job_id):
    """Fetches all Textract pages for a job until complete."""
    pages = []
    next_token = None
    page_count = 0

    while True:
        if next_token:
            response = safe_textract_call(
                textract.get_document_text_detection,
                JobId=job_id,
                NextToken=next_token
            )
        else:
            response = safe_textract_call(
                textract.get_document_text_detection,
                JobId=job_id
            )

        blocks = response.get("Blocks", [])
        pages.extend(blocks)
        page_count += 1
        logger.info(f"📄 Retrieved page {page_count} ({len(blocks)} blocks)")

        next_token = response.get("NextToken")
        if not next_token:
            break
        time.sleep(1)

    logger.info(f"✅ Total blocks retrieved: {len(pages)} across {page_count} pages")
    return {"Blocks": pages}

# ---------------------------------------------------------------------
# 🧠 Textract single-PDF helper
# ---------------------------------------------------------------------
def run_textract_for_pdf(pdf_key):
    """
    Runs Textract OCR for one PDF in S3 and saves output locally.
    Accepts either string or Path for pdf_key.
    Returns string path of saved JSON.
    """
    # Handle string vs Path
    if not isinstance(pdf_key, Path):
        pdf_key = Path(pdf_key)

    task_name = f"run_textract_{pdf_key.stem}"
    track_task(task_name, "STARTED")

    try:
        logger.info(f"🚀 Starting Textract for: {pdf_key}")
        start_response = safe_textract_call(
            textract.start_document_text_detection,
            DocumentLocation={"S3Object": {"Bucket": BUCKET, "Name": str(pdf_key)}},
        )
        job_id = start_response["JobId"]
        logger.info(f"🪄 Job started: {job_id}")

        # Poll for completion
        while True:
            result = safe_textract_call(textract.get_document_text_detection, JobId=job_id)
            status = result["JobStatus"]
            if status in ["SUCCEEDED", "FAILED"]:
                break
            logger.info(f"⏳ Waiting for Textract job to finish for {pdf_key}…")
            time.sleep(5)

        if status == "SUCCEEDED":
            full_result = get_all_textract_results(job_id)
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            out_path = RAW_DIR / f"{pdf_key.stem}_raw.json"

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(full_result, f, indent=2)

            msg = f"✅ Saved Textract output: {out_path} ({len(full_result['Blocks'])} blocks)"
            logger.info(msg)
            track_task(task_name, "SUCCESS", details=msg)
            return str(out_path)

        else:
            msg = f"❌ Textract failed for {pdf_key}"
            logger.error(msg)
            track_task(task_name, "FAILED", error=msg)
            return None

    except Exception as e:
        err = f"❌ Error running Textract for {pdf_key}: {e}"
        logger.exception(err)
        track_task(task_name, "FAILED", error=str(e))
        return None

# ---------------------------------------------------------------------
# 🧩 Batch Textract Runner
# ---------------------------------------------------------------------
def run_textract_all(**context):
    """
    Runs Textract OCR on all PDFs fetched from S3.

    - Skips already processed PDFs
    - Launches new Textract jobs in parallel
    - Polls until all jobs complete
    Returns list of string paths for new raw JSON outputs.
    """
    task_name = "run_textract_all"
    track_task(task_name, "STARTED")

    pdfs = fetch_files()
    if not pdfs:
        msg = "⚠️ No PDFs found in S3 bucket."
        logger.warning(msg)
        track_task(task_name, "SUCCESS", details=msg)
        return []

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    output_files = []

    # 1️⃣ Start Textract jobs for new PDFs
    for pdf_key in pdfs:
        raw_path = RAW_DIR / f"{Path(pdf_key).stem}_raw.json"
        if raw_path.exists():
            logger.info(f"⚡ Skipping {pdf_key} (already processed).")
            continue

        per_file_task = f"run_textract_{Path(pdf_key).stem}"
        track_task(per_file_task, "STARTED")

        try:
            start_response = safe_textract_call(
                textract.start_document_text_detection,
                DocumentLocation={"S3Object": {"Bucket": BUCKET, "Name": pdf_key}},
            )
            job_id = start_response["JobId"]
            logger.info(f"🪄 Started Textract job for {pdf_key}: {job_id}")
            jobs.append({"pdf_key": pdf_key, "job_id": job_id, "task_name": per_file_task})
        except Exception as e:
            err = f"❌ Error starting Textract for {pdf_key}: {e}"
            logger.exception(err)
            track_task(per_file_task, "FAILED", error=str(e))

    if not jobs:
        msg = "⚠️ No new PDFs to process."
        logger.warning(msg)
        track_task(task_name, "SUCCESS", details=msg)
        return output_files

    # 2️⃣ Poll until all jobs complete
    POLL_INTERVAL = 5
    while jobs:
        for job in list(jobs):  # iterate copy
            pdf_key = job["pdf_key"]
            job_id = job["job_id"]
            per_file_task = job["task_name"]

            try:
                result = safe_textract_call(textract.get_document_text_detection, JobId=job_id)
                status = result.get("JobStatus")
            except Exception as e:
                err = f"❌ Error checking Textract status for {pdf_key}: {e}"
                logger.exception(err)
                track_task(per_file_task, "FAILED", error=str(e))
                jobs.remove(job)
                continue

            if status == "SUCCEEDED":
                full_result = get_all_textract_results(job_id)
                out_path = RAW_DIR / f"{Path(pdf_key).stem}_raw.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(full_result, f, indent=2)

                msg = f"✅ Saved Textract output: {out_path}"
                logger.info(msg)
                track_task(per_file_task, "SUCCESS", details=msg)
                output_files.append(str(out_path))
                jobs.remove(job)

            elif status == "FAILED":
                msg = f"❌ Textract failed for {pdf_key} ({job_id})"
                logger.error(msg)
                track_task(per_file_task, "FAILED", error=msg)
                jobs.remove(job)
            else:
                logger.info(f"⏳ Waiting for Textract job to finish for {pdf_key} (status={status})")

        time.sleep(POLL_INTERVAL)

    msg = f"✅ Completed Textract for {len(output_files)} files."
    logger.info(msg)
    track_task(task_name, "SUCCESS", details=msg)
    return output_files

# ---------------------------------------------------------------------
# 🏁 Entry point (manual run)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    run_textract_all()
