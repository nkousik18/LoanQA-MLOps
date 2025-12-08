"""
segment_text.py
---------------
Stage 3: Segments raw Textract JSON files into line-level spans.
Each span has fields like doc_id, page, text, confidence, and bounding box.

Features:
- Reads from data/aws_extraction_data/raw/
- Writes segmented JSONs to data/aws_extraction_data/segmented/
- Logs and tracks every file processed
- Compatible with both VS Code (local) and Airflow (Docker)
- GCS-aware (uses gcs_utils + config paths)
"""

import os
import sys
import json
import uuid
from pathlib import Path
from typing import List

from tqdm import tqdm
from google.cloud import storage

# ---------------------------------------------------------------------
# Ensure working directory and imports
# ---------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# ---------------------------------------------------------------------
# Imports from central config and utilities
# ---------------------------------------------------------------------
from scripts.aws_extraction_scripts.config import (
    RAW_DIR,
    SEGMENTED_DIR,
    GCS_BUCKET,
    USE_GCS_OUTPUT,
    to_gcs_key,
)
from scripts.aws_extraction_scripts.log_utils import get_logger
from scripts.aws_extraction_scripts.tracker import track_task
from scripts.aws_extraction_scripts.gcs_utils import (
    read_json,
    write_json,
    logical_exists,
)

# ---------------------------------------------------------------------
# Initialize logger
# ---------------------------------------------------------------------
logger = get_logger("segment_text")


# ---------------------------------------------------------------------
# Core: Segment a single Textract JSON file into a target directory
# ---------------------------------------------------------------------
def segment_textract_json_to_dir(raw_path, segmented_dir):
    """
    Converts a Textract raw JSON into line-level segmented format.

    - raw_path: str or Path to *_raw.json (logical path under RAW_DIR)
    - segmented_dir: directory (Path-like) where segmented file should be written

    Uses GCS-aware read/write helpers, so storage can be GCS or local.
    Returns output file path as string, or None on failure.
    """
    raw_path = Path(raw_path)
    segmented_dir = Path(segmented_dir)

    task_name = f"segment_{raw_path.stem}"
    track_task(task_name, "STARTED")

    try:
        if not logical_exists(raw_path):
            raise FileNotFoundError(f"Raw file not found (GCS/local): {raw_path}")

        # Read Textract JSON from GCS or local
        data = read_json(raw_path)
        blocks = data.get("Blocks", [])

        doc_id = str(uuid.uuid4())
        segmented = []
        span_id = 1

        logger.info(f"Processing file: {raw_path}")
        logger.info(f"Total blocks found: {len(blocks)}")

        for block in blocks:
            if block.get("BlockType") == "LINE":
                segmented.append(
                    {
                        "doc_id": doc_id,
                        "page": block.get("Page", 1),
                        "span_id": span_id,
                        "span_type": "line",
                        "text": block.get("Text", "").strip(),
                        "conf": block.get("Confidence", 0),
                        "bbox": block.get("Geometry", {}).get("BoundingBox", {}),
                    }
                )
                span_id += 1

        # Logical output path (same naming pattern, but under segmented_dir)
        out_path = segmented_dir / f"{raw_path.stem.replace('_raw', '_segmented')}.json"

        # Write segmented JSON (GCS/local depending on config)
        write_json(out_path, segmented)

        msg = f"Segmented output saved: {out_path} (Lines: {len(segmented)})"
        logger.info(msg)
        track_task(task_name, "SUCCESS", details=msg)
        return str(out_path)

    except Exception as e:
        err = f"Error segmenting {raw_path}: {e}"
        logger.exception(err)
        track_task(task_name, "FAILED", error=str(e))
        return None


# ---------------------------------------------------------------------
# Backwards-compatible single-file function (uses SEGMENTED_DIR)
# ---------------------------------------------------------------------
def segment_textract_json(raw_path):
    """
    Original function signature, kept for batch/Airflow.

    Uses global SEGMENTED_DIR from config.
    """
    return segment_textract_json_to_dir(raw_path, SEGMENTED_DIR)


# ---------------------------------------------------------------------
# Helpers: list raw files from GCS or local
# ---------------------------------------------------------------------
def _list_raw_files_gcs() -> List[Path]:
    """
    List all *_raw.json logical paths under RAW_DIR in GCS.
    """
    client = storage.Client()
    prefix = to_gcs_key(RAW_DIR)
    if not prefix.endswith("/"):
        prefix += "/"

    logger.info(
        f"[GCS] Listing raw Textract JSONs in bucket '{GCS_BUCKET}' "
        f"under prefix '{prefix}'"
    )

    blobs = client.list_blobs(GCS_BUCKET, prefix=prefix)
    raw_paths: List[Path] = []

    for blob in blobs:
        name = blob.name
        if not name.endswith("_raw.json"):
            continue
        # name looks like "<prefix>Something_raw.json"
        rel = name[len(prefix) :]
        if not rel or rel.endswith("/"):
            continue
        raw_paths.append(RAW_DIR / rel)

    logger.info(f"[GCS] Found {len(raw_paths)} raw JSON files.")
    return raw_paths


def _list_raw_files_local() -> List[Path]:
    """
    List all *_raw.json files from local RAW_DIR.
    """
    if not RAW_DIR.exists():
        logger.warning(f"RAW_DIR not found: {RAW_DIR}")
        return []
    raw_files = list(RAW_DIR.glob("*_raw.json"))
    logger.info(f"[LOCAL] Found {len(raw_files)} raw JSON files in {RAW_DIR}")
    return raw_files


# ---------------------------------------------------------------------
# Run segmentation for all raw files (batch)
# ---------------------------------------------------------------------
def run_segmentation_all(**context):
    """
    Segments all Textract raw JSON files in data/aws_extraction_data/raw/.

    Uses:
      - GCS listing when USE_GCS_OUTPUT=True
      - Local filesystem listing otherwise

    Returns list of string paths for segmented JSONs.
    """
    task_name = "run_segmentation_all"
    track_task(task_name, "STARTED")

    # Decide where to list from
    if USE_GCS_OUTPUT:
        raw_files = _list_raw_files_gcs()
    else:
        raw_files = _list_raw_files_local()

    if not raw_files:
        msg = f"No raw JSON files found for segmentation."
        logger.warning(msg)
        track_task(task_name, "FAILED", error=msg)
        return []

    results = []
    for raw_path in tqdm(raw_files, desc="Segmenting Textract JSONs"):
        out_file = segment_textract_json(raw_path)
        if out_file:
            results.append(out_file)

    msg = f"Completed segmentation for {len(results)} files."
    logger.info(msg)
    track_task(task_name, "SUCCESS", details=msg)
    return results


# ---------------------------------------------------------------------
# Entry point for standalone run
# ---------------------------------------------------------------------
if __name__ == "__main__":
    run_segmentation_all()
