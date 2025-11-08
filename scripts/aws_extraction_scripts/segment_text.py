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
"""

import os
import sys
import json
import uuid
from tqdm import tqdm
from pathlib import Path

# ---------------------------------------------------------------------
# 🔧 Ensure working directory and imports
# ---------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# ---------------------------------------------------------------------
# 📦 Imports from central config and utilities
# ---------------------------------------------------------------------
from scripts.aws_extraction_scripts.config import RAW_DIR, SEGMENTED_DIR
from scripts.aws_extraction_scripts.log_utils import get_logger
from scripts.aws_extraction_scripts.tracker import track_task

# ---------------------------------------------------------------------
# ⚙️ Initialize logger
# ---------------------------------------------------------------------
logger = get_logger("segment_text")

# ---------------------------------------------------------------------
# 🧩 Core: Segment a single Textract JSON file
# ---------------------------------------------------------------------
def segment_textract_json(raw_path):
    """
    Converts a Textract raw JSON into line-level segmented format.
    Accepts either str or Path input.
    Returns output file path as string (Airflow-safe).
    """

    # ✅ Handle both str and Path input (important for Airflow + pytest)
    raw_path = Path(raw_path)

    task_name = f"segment_{raw_path.stem}"
    track_task(task_name, "STARTED")

    try:
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw file not found: {raw_path}")

        with open(raw_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        blocks = data.get("Blocks", [])
        doc_id = str(uuid.uuid4())
        segmented = []
        span_id = 1

        logger.info(f"📂 Processing file: {raw_path}")
        logger.info(f"📦 Total blocks found: {len(blocks)}")

        for block in blocks:
            if block.get("BlockType") == "LINE":
                segmented.append({
                    "doc_id": doc_id,
                    "page": block.get("Page", 1),
                    "span_id": span_id,
                    "span_type": "line",
                    "text": block.get("Text", "").strip(),
                    "conf": block.get("Confidence", 0),
                    "bbox": block.get("Geometry", {}).get("BoundingBox", {}),
                })
                span_id += 1

        SEGMENTED_DIR.mkdir(parents=True, exist_ok=True)
        out_path = SEGMENTED_DIR / f"{raw_path.stem.replace('_raw', '_segmented')}.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(segmented, f, indent=2, ensure_ascii=False)

        msg = f"✅ Segmented output saved: {out_path} (Lines: {len(segmented)})"
        logger.info(msg)
        track_task(task_name, "SUCCESS", details=msg)
        return str(out_path)

    except Exception as e:
        err = f"❌ Error segmenting {raw_path}: {e}"
        logger.exception(err)
        track_task(task_name, "FAILED", error=str(e))
        return None


# ---------------------------------------------------------------------
# 🚀 Run segmentation for all raw files
# ---------------------------------------------------------------------
def run_segmentation_all(**context):
    """
    Segments all Textract raw JSON files in data/aws_extraction_data/raw/.
    Returns list of string paths for segmented JSONs.
    """
    task_name = "run_segmentation_all"
    track_task(task_name, "STARTED")

    if not RAW_DIR.exists():
        msg = f"⚠️ RAW_DIR not found: {RAW_DIR}"
        logger.warning(msg)
        track_task(task_name, "FAILED", error=msg)
        return []

    raw_files = list(RAW_DIR.glob("*_raw.json"))
    if not raw_files:
        msg = f"⚠️ No raw JSON files found in {RAW_DIR}"
        logger.warning(msg)
        track_task(task_name, "FAILED", error=msg)
        return []

    results = []
    for raw_path in tqdm(raw_files, desc="Segmenting Textract JSONs"):
        out_file = segment_textract_json(raw_path)
        if out_file:
            results.append(out_file)

    msg = f"✅ Completed segmentation for {len(results)} files."
    logger.info(msg)
    track_task(task_name, "SUCCESS", details=msg)
    return results


# ---------------------------------------------------------------------
# 🏁 Entry point for standalone run
# ---------------------------------------------------------------------
if __name__ == "__main__":
    run_segmentation_all()
