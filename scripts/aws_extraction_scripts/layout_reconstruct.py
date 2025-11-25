"""
layout_reconstruct.py
---------------------
Stage X: Reconstructs human-readable layout text from normalized JSON spans.

Input:
  - data/aws_extraction_data/normalized/*_normalized.json

Output:
  - data/aws_extraction_data/layout_reconstructed/*_normalized_layout.txt

Features:
  - Uses normalized JSON (page, bbox, text_preserved/text_display)
  - Sorts spans into reading order (by page, then y, then x)
  - Batch mode (all normalized files) + single-file helper
  - Same logging + tracker pattern as other AWS extraction scripts
"""

import os
import sys
import json
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
from scripts.aws_extraction_scripts.config import NORMALIZED_DIR, LAYOUT_DIR
from scripts.aws_extraction_scripts.log_utils import get_logger
from scripts.aws_extraction_scripts.tracker import track_task

logger = get_logger("layout_reconstruct")


# ---------------------------------------------------------------------
# Utility: sort spans in global reading order
# ---------------------------------------------------------------------
def sort_spans_reading_order(spans, y_tol: float = 0.008, x_tol: float = 0.003):
    """
    Sort spans by approximate reading order using bbox:

      1. Sort by top (y), then left (x)
      2. Group into 'lines' using y tolerance
      3. Sort each line by x and join text_preserved

    Assumes each span has:
      - "bbox" with keys "Top" and "Left"
      - "text_preserved" (or falls back to "text_display"/"text")
    """
    if not spans:
        return []

    # 1) Sort by Y, then X
    spans_sorted = sorted(
        spans,
        key=lambda s: (
            s.get("bbox", {}).get("Top", 0.0),
            s.get("bbox", {}).get("Left", 0.0),
        ),
    )

    # 2) Group into lines based on Y tolerance
    lines = []
    current_line = [spans_sorted[0]]
    prev_y = spans_sorted[0].get("bbox", {}).get("Top", 0.0)

    for s in spans_sorted[1:]:
        y = s.get("bbox", {}).get("Top", 0.0)
        if abs(y - prev_y) <= y_tol:
            current_line.append(s)
        else:
            lines.append(current_line)
            current_line = [s]
        prev_y = y

    lines.append(current_line)

    # 3) Sort each line by X and join text
    clean_lines = []
    for line in lines:
        line_sorted = sorted(
            line,
            key=lambda t: t.get("bbox", {}).get("Left", 0.0),
        )
        parts = []
        for t in line_sorted:
            txt = (
                t.get("text_preserved")
                or t.get("text_display")
                or t.get("text", "")
            )
            if txt:
                parts.append(str(txt).strip())
        if parts:
            clean_lines.append(" ".join(parts))

    return clean_lines


# ---------------------------------------------------------------------
# Reconstruct entire document from normalized spans
# ---------------------------------------------------------------------
def reconstruct_document(spans):
    """
    Given a list of normalized spans (for all pages), build a readable text:

      === PAGE 1 ===
      line ...
      line ...

    Returns a single string.
    """
    pages = {}
    for s in spans:
        page = s.get("page", 1)
        pages.setdefault(page, []).append(s)

    output_lines = []

    for page in sorted(pages.keys()):
        output_lines.append(f"\n=== PAGE {page} ===\n")
        page_spans = pages[page]
        page_lines = sort_spans_reading_order(page_spans)
        output_lines.extend(page_lines)
        output_lines.append("")  # blank line between pages

    return "\n".join(output_lines)


# ---------------------------------------------------------------------
# Core: reconstruct layout for ONE normalized JSON into a target dir
# ---------------------------------------------------------------------
def reconstruct_layout_to_dir(normalized_path, layout_dir) -> str | None:
    """
    normalized_path: path to *_normalized.json
    layout_dir: directory where *_layout.txt should be written

    Returns the layout file path (str) or None on failure.
    """
    normalized_path = Path(normalized_path)
    layout_dir = Path(layout_dir)

    task_name = f"layout_{normalized_path.stem}"
    track_task(task_name, "STARTED")

    try:
        if not normalized_path.exists():
            raise FileNotFoundError(f"Normalized file not found: {normalized_path}")

        with open(normalized_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Normalized files are a list of span dicts
        spans = data["spans"] if isinstance(data, dict) and "spans" in data else data

        layout_dir.mkdir(parents=True, exist_ok=True)

        base = normalized_path.stem  # e.g. loan_001_normalized
        out_path = layout_dir / f"{base}_layout.txt"

        layout_text = reconstruct_document(spans)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(layout_text)

        msg = f"Layout reconstruction complete: {out_path}"
        logger.info(msg)
        track_task(task_name, "SUCCESS", details=msg)
        return str(out_path)

    except Exception as e:
        err = f"Error during layout reconstruction for {normalized_path}: {e}"
        logger.exception(err)
        track_task(task_name, "FAILED", error=str(e))
        return None


# ---------------------------------------------------------------------
# Backwards-compatible helper: use global LAYOUT_DIR
# ---------------------------------------------------------------------
def reconstruct_layout(normalized_path) -> str | None:
    """
    Wrapper for batch / other stages.
    Writes into config.LAYOUT_DIR.
    """
    return reconstruct_layout_to_dir(normalized_path, LAYOUT_DIR)


# ---------------------------------------------------------------------
# Batch runner: run on ALL normalized JSON files
# ---------------------------------------------------------------------
def run_layout_all(**context):
    """
    Runs layout reconstruction for all *_normalized.json files in
    data/aws_extraction_data/normalized/.

    Returns list of layout file paths.
    """
    task_name = "run_layout_all"
    track_task(task_name, "STARTED")

    if not NORMALIZED_DIR.exists():
        msg = f"NORMALIZED_DIR not found: {NORMALIZED_DIR}"
        logger.warning(msg)
        track_task(task_name, "FAILED", error=msg)
        return []

    norm_files = list(NORMALIZED_DIR.glob("*_normalized.json"))
    if not norm_files:
        msg = f"No normalized JSON files found in {NORMALIZED_DIR}"
        logger.warning(msg)
        track_task(task_name, "FAILED", error=msg)
        return []

    results = []
    for norm_path in norm_files:
        out_file = reconstruct_layout(norm_path)
        if out_file:
            results.append(out_file)

    msg = f"Completed layout reconstruction for {len(results)} files."
    logger.info(msg)
    track_task(task_name, "SUCCESS", details=msg)
    return results


# ---------------------------------------------------------------------
# 🏁 Manual run (VS Code / local)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Simple batch run over all *_normalized.json
    run_layout_all()
