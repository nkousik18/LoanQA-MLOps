"""
run_textract.py
---------------
Stage 2: Runs AWS Textract OCR on PDFs from S3 and saves raw JSON results.
Includes PII Masking for Loan Documents.
"""

import os
import sys
import json
import time
import boto3
import botocore
from pathlib import Path

# ---------------------------------------------------------------------
# Ensure project root dynamically (works in both local & Docker)
# ---------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# ---------------------------------------------------------------------
# Centralized imports
# ---------------------------------------------------------------------
from scripts.aws_extraction_scripts.config import (
    BUCKET,
    REGION,
    RAW_DIR,
    RAW_TEXT_DIR,
)
from scripts.aws_extraction_scripts.log_utils import get_logger
from scripts.aws_extraction_scripts.tracker import track_task
from scripts.aws_extraction_scripts.fetch_files import fetch_files, DOC_PREFIX

### PII MODIFICATION: Import the masker ###
from scripts.aws_extraction_scripts.pii_masking import PIIMasker

# ---------------------------------------------------------------------
# Initialize clients, logger, and PII Masker
# ---------------------------------------------------------------------
logger = get_logger("run_textract")
s3 = boto3.client("s3", region_name=REGION)
textract = boto3.client("textract", region_name=REGION)
pii_masker = PIIMasker()  # Initialize PII Masker


# ---------------------------------------------------------------------
# Helper: convert Textract blocks to plain text
# ---------------------------------------------------------------------
def textract_blocks_to_text(blocks):
    """
    Convert Textract 'Blocks' list into a multiline string.
    """
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
def safe_textract_call(func, **kwargs):
    for attempt in range(5):
        try:
            return func(**kwargs)
        except (botocore.exceptions.EndpointConnectionError, botocore.exceptions.SSLError) as e:
            logger.warning(f"Network issue, retrying in 5s: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error, retrying in 5s: {e}")
        time.sleep(5)
    raise Exception("Failed to connect to Textract after multiple attempts.")


# ---------------------------------------------------------------------
# Fetch all Textract results for a job
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
# Textract single-PDF helper (path-aware, for sessions)
# ---------------------------------------------------------------------
def run_textract_for_pdf_to_dirs(pdf_key, raw_dir, raw_text_dir):
    # Normalize to an S3 key with forward slashes
    if isinstance(pdf_key, Path):
        s3_key = str(pdf_key).replace("\\", "/")
    else:
        s3_key = str(pdf_key).replace("\\", "/")
        pdf_key = Path(s3_key)

    raw_dir = Path(raw_dir)
    raw_text_dir = Path(raw_text_dir)

    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_text_dir.mkdir(parents=True, exist_ok=True)

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

        ### PII MODIFICATION START ###
        # Mask PII in the blocks BEFORE saving or converting to text
        blocks = pii_masker.mask_textract_blocks(blocks)

        # Update full_result with masked blocks so JSON is safe
        full_result["Blocks"] = blocks
        ### PII MODIFICATION END ###

        json_path = raw_dir / f"{pdf_key.stem}_raw.json"
        txt_path = raw_text_dir / f"{pdf_key.stem}_raw.txt"

        # JSON (Now Safe)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(full_result, f, indent=2)

        # TXT (Generated from Safe Blocks)
        text_content = textract_blocks_to_text(blocks)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text_content)

        msg = f"Saved masked JSON: {json_path} and text: {txt_path}"
        logger.info(msg)
        track_task(task_name, "SUCCESS", details=msg)
        return str(json_path), str(txt_path)

    except Exception as e:
        err = f"Error running Textract for {s3_key}: {e}"
        logger.exception(err)
        track_task(task_name, "FAILED", error=str(e))
        return None, None


def run_textract_for_pdf(pdf_key):
    json_path, _ = run_textract_for_pdf_to_dirs(pdf_key, RAW_DIR, RAW_TEXT_DIR)
    return json_path


def run_textract_all(**context):
    task_name = "run_textract_all"
    track_task(task_name, "STARTED")

    pdfs = fetch_files(prefix=DOC_PREFIX)
    if not pdfs:
        return []

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    RAW_TEXT_DIR.mkdir(parents=True, exist_ok=True)

    jobs = []
    output_files = []

    for pdf_key in pdfs:
        pdf_stem = Path(pdf_key).stem
        json_path = RAW_DIR / f"{pdf_stem}_raw.json"
        txt_path = RAW_TEXT_DIR / f"{pdf_stem}_raw.txt"

        if json_path.exists() and txt_path.exists():
            output_files.append(str(json_path))
            continue

        if json_path.exists() and not txt_path.exists():
            # If rebuilding TXT from JSON, ensure JSON is masked
            # (Assuming existing JSON is already masked if this script ran before.
            # If not, you might want to re-mask here just in case).
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                blocks = data.get("Blocks", [])

                ### PII RE-CHECK (Optional but recommended) ###
                blocks = pii_masker.mask_textract_blocks(blocks)
                ### END PII ###

                text_content = textract_blocks_to_text(blocks)
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(text_content)
                output_files.append(str(json_path))
            except Exception:
                continue
            continue

        # Start Textract Job
        per_file_task = f"run_textract_{pdf_stem}"
        try:
            start_response = safe_textract_call(
                textract.start_document_text_detection,
                DocumentLocation={"S3Object": {"Bucket": BUCKET, "Name": pdf_key}},
            )
            jobs.append({
                "pdf_key": pdf_key,
                "pdf_stem": pdf_stem,
                "job_id": start_response["JobId"],
                "task_name": per_file_task
            })
        except Exception as e:
            track_task(per_file_task, "FAILED", error=str(e))

    # Poll jobs
    POLL_INTERVAL = 5
    while jobs:
        for job in list(jobs):
            try:
                result = safe_textract_call(
                    textract.get_document_text_detection,
                    JobId=job["job_id"],
                )
                status = result.get("JobStatus")
            except Exception:
                jobs.remove(job)
                continue

            if status == "SUCCEEDED":
                full_result = get_all_textract_results(job["job_id"])
                blocks = full_result.get("Blocks", [])

                ### PII MODIFICATION ###
                blocks = pii_masker.mask_textract_blocks(blocks)
                full_result["Blocks"] = blocks
                ### END PII ###

                json_path = RAW_DIR / f"{job['pdf_stem']}_raw.json"
                txt_path = RAW_TEXT_DIR / f"{job['pdf_stem']}_raw.txt"

                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(full_result, f, indent=2)

                text_content = textract_blocks_to_text(blocks)
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(text_content)

                output_files.append(str(json_path))
                jobs.remove(job)
            elif status == "FAILED":
                jobs.remove(job)

            time.sleep(1)  # Small throttle inside loop
        time.sleep(POLL_INTERVAL)

    track_task(task_name, "SUCCESS")
    return output_files


if __name__ == "__main__":
    run_textract_all()