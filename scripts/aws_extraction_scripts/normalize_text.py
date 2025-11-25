"""
normalize_text.py
-----------------
Stage 4: Normalizes segmented JSON files by adding three text variants:
 - text_display: original readable form
 - text_clean: aggressively cleaned for embeddings/search
 - text_preserved: lightly cleaned for entity extraction

Features:
- Reads from data/aws_extraction_data/segmented/
- Writes to data/aws_extraction_data/normalized/
- Tracks progress in manifest.json
- Logs all operations (local & Airflow compatible)
"""

import os
import sys
import json
import re
from tqdm import tqdm
from pathlib import Path

# ---------------------------------------------------------------------
# Ensure working directory and imports
# ---------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# ---------------------------------------------------------------------
# Imports from config and utilities
# ---------------------------------------------------------------------
from scripts.aws_extraction_scripts.config import SEGMENTED_DIR, NORMALIZED_DIR
from scripts.aws_extraction_scripts.log_utils import get_logger
from scripts.aws_extraction_scripts.tracker import track_task

logger = get_logger("normalize_text")


# ---------------------------------------------------------------------
# Cleaning Helpers
# ---------------------------------------------------------------------
def clean_text_basic(text: str) -> str:
    """Aggressively cleans text for embeddings/search."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text_preserved(text: str) -> str:
    """Lightly cleans text while preserving key symbols for entity extraction."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9$%@:;.,()/\-_\\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------
# Core Function: Normalize one segmented file into target directory
# ---------------------------------------------------------------------
def normalize_segmented_json_to_dir(seg_path, normalized_dir):
    """
    Converts a segmented JSON file into normalized form with
    text_display, text_clean, and text_preserved variants.

    - seg_path: str or Path to *_segmented.json
    - normalized_dir: target directory for *_normalized.json

    Returns output path as string.
    """
    seg_path = Path(seg_path)
    normalized_dir = Path(normalized_dir)

    task_name = f"normalize_{seg_path.stem}"
    track_task(task_name, "STARTED")

    try:
        if not seg_path.exists():
            raise FileNotFoundError(f"Segmented file not found: {seg_path}")

        with open(seg_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        normalized = []
        for entry in data:
            raw_text = entry.get("text", "")
            norm_entry = entry.copy()
            norm_entry["text_display"] = raw_text.strip()
            norm_entry["text_clean"] = clean_text_basic(raw_text)
            norm_entry["text_preserved"] = clean_text_preserved(raw_text)
            normalized.append(norm_entry)

        normalized_dir.mkdir(parents=True, exist_ok=True)
        out_path = normalized_dir / f"{seg_path.stem.replace('_segmented', '_normalized')}.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(normalized, f, indent=2, ensure_ascii=False)

        msg = f"Normalized file saved: {out_path} (Lines: {len(normalized)})"
        logger.info(msg)
        track_task(task_name, "SUCCESS", details=msg)
        return str(out_path)

    except Exception as e:
        err = f"Error normalizing {seg_path}: {e}"
        logger.exception(err)
        track_task(task_name, "FAILED", error=str(e))
        return None


# ---------------------------------------------------------------------
# Backwards-compatible function (uses NORMALIZED_DIR)
# ---------------------------------------------------------------------
def normalize_segmented_json(seg_path):
    """
    Original function signature, kept for batch/Airflow.

    Uses global NORMALIZED_DIR from config.
    """
    return normalize_segmented_json_to_dir(seg_path, NORMALIZED_DIR)


# ---------------------------------------------------------------------
# Batch Runner
# ---------------------------------------------------------------------
def run_normalization_all(**context):
    """
    Runs normalization for all segmented JSON files.
    Returns list of string paths for normalized JSONs.
    """
    task_name = "run_normalization_all"
    track_task(task_name, "STARTED")

    if not SEGMENTED_DIR.exists():
        msg = f"Segmented directory not found: {SEGMENTED_DIR}"
        logger.warning(msg)
        track_task(task_name, "FAILED", error=msg)
        return []

    seg_files = list(SEGMENTED_DIR.glob("*_segmented.json"))
    if not seg_files:
        msg = f"No segmented JSON files found in {SEGMENTED_DIR}"
        logger.warning(msg)
        track_task(task_name, "FAILED", error=msg)
        return []

    results = []
    for seg_path in tqdm(seg_files, desc="Normalizing segmented JSONs"):
        out_file = normalize_segmented_json(seg_path)
        if out_file:
            results.append(out_file)

    msg = f"Completed normalization for {len(results)} files."
    logger.info(msg)
    track_task(task_name, "SUCCESS", details=msg)
    return results


# ---------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    run_normalization_all()
