"""
single_pdf_pipeline.py
----------------------
Single-PDF pipeline for live user uploads.

Stages:
  1. Textract OCR  -> raw JSON  + raw text
  2. Segmentation  -> line-level spans
  3. Normalization -> text_display / text_clean / text_preserved
  4. Layout        -> human-readable layout text from normalized spans
  5. Session RAG   -> build chunks/blocks/embeddings under session/rag/

Outputs are stored under a per-session folder:
  data/local_pipeline/sessions/<session_id>/{raw, raw_text, segmented, normalized, layout, rag}
"""

from __future__ import annotations

import os
import sys
import uuid
import json
from pathlib import Path

# ---------------------------------------------------------------------
# 🔧 Ensure project root on sys.path  (same pattern as other scripts)
# ---------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))             # .../scripts/aws_extraction_scripts
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))  # .../doc-understand

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.chdir(PROJECT_ROOT)

# ---------------------------------------------------------------------
# 📦 Absolute imports (NO relative "from .")
# ---------------------------------------------------------------------
from scripts.aws_extraction_scripts.run_textract import run_textract_for_pdf_to_dirs
from scripts.aws_extraction_scripts.segment_text import segment_textract_json_to_dir
from scripts.aws_extraction_scripts.normalize_text import normalize_segmented_json_to_dir
from scripts.aws_extraction_scripts.layout_reconstruct import reconstruct_layout_to_dir
from scripts.aws_extraction_scripts.log_utils import get_logger

# ✅ Session RAG builder now under LLM/forms_llm
from scripts.LLM.forms_llm.rag_reasoning_scripts.session_rag_builder import (
    build_session_rag_from_normalized,
)

logger = get_logger("single_pdf_pipeline")

# Root for all live sessions
SESSIONS_ROOT = Path(PROJECT_ROOT) / "data" / "local_pipeline" / "sessions"


# ---------------------------------------------------------------------
# 📁 Session directory helper
# ---------------------------------------------------------------------
def make_session_dirs(session_id: str) -> dict:
    """
    Create per-session directories and return all paths.
    """
    session_root = SESSIONS_ROOT / session_id
    raw_dir = session_root / "raw"
    raw_text_dir = session_root / "raw_text"
    segmented_dir = session_root / "segmented"
    normalized_dir = session_root / "normalized"
    layout_dir = session_root / "layout"   # for layout reconstruction
    rag_dir = session_root / "rag"         # ✅ executor expects this

    for d in (raw_dir, raw_text_dir, segmented_dir, normalized_dir, layout_dir, rag_dir):
        d.mkdir(parents=True, exist_ok=True)

    return {
        "session_id": session_id,
        "session_root": str(session_root),
        "raw_dir": raw_dir,
        "raw_text_dir": raw_text_dir,
        "segmented_dir": segmented_dir,
        "normalized_dir": normalized_dir,
        "layout_dir": layout_dir,
        "rag_dir": rag_dir,
    }


# ---------------------------------------------------------------------
# 🚀 Main single-PDF pipeline
# ---------------------------------------------------------------------
def process_single_pdf_session(
    pdf_s3_key: str,
    session_id: str | None = None
) -> dict:
    """
    Run the full OCR → segmentation → normalization → layout → session RAG pipeline for ONE PDF.

    Args:
        pdf_s3_key: S3 key, e.g. "user_uploads/loan1.pdf"
        session_id: optional fixed session id; if None, a random one is created.

    Returns:
        dict with paths for all outputs (raw_json, raw_text, segmented, normalized, layout, rag).
    """
    if session_id is None:
        session_id = f"session_{uuid.uuid4().hex[:8]}"

    dirs = make_session_dirs(session_id)

    # --- Step 1: Textract OCR ---
    raw_json_path, raw_text_path = run_textract_for_pdf_to_dirs(
        pdf_s3_key,
        dirs["raw_dir"],
        dirs["raw_text_dir"],
    )
    if raw_json_path is None:
        raise RuntimeError(f"Textract failed for {pdf_s3_key}")

    # --- Step 2: Segmentation ---
    segmented_path = segment_textract_json_to_dir(
        raw_json_path,
        dirs["segmented_dir"],
    )
    if segmented_path is None:
        raise RuntimeError(f"Segmentation failed for {raw_json_path}")

    # --- Step 3: Normalization ---
    normalized_path = normalize_segmented_json_to_dir(
        segmented_path,
        dirs["normalized_dir"],
    )
    if normalized_path is None:
        raise RuntimeError(f"Normalization failed for {segmented_path}")

    # --- Step 4: Layout reconstruction (from normalized JSON) ---
    layout_path = reconstruct_layout_to_dir(
        normalized_path,
        dirs["layout_dir"],
    )
    if layout_path is None:
        raise RuntimeError(f"Layout reconstruction failed for {normalized_path}")

    # --- Step 5: Build Session RAG (THIS is the missing piece for UI) ---
    rag_info = build_session_rag_from_normalized(str(normalized_path))
    # rag_info already contains rag_dir, chunks, blocks, embeddings paths

    info = {
        "session_id": session_id,
        "session_root": dirs["session_root"],
        "raw_json": raw_json_path,
        "raw_text": raw_text_path,
        "segmented": segmented_path,
        "normalized": normalized_path,
        "layout": layout_path,
        "rag": rag_info.get("rag_dir"),
        "rag_chunks": rag_info.get("chunks_path"),
        "rag_blocks": rag_info.get("blocks_path"),
        "rag_embeddings": rag_info.get("embeddings_path"),
    }

    logger.info("✅ Single PDF session complete:\n%s", json.dumps(info, indent=2))
    return info


# ---------------------------------------------------------------------
# 🧪 Quick manual test (from VS Code terminal)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    test_pdf_key = "user_uploads/Personal-Loan-Agreement - Copy.pdf"
    info = process_single_pdf_session(test_pdf_key)
    print("\n=== Single PDF Session Info ===")
    for k, v in info.items():
        print(f"{k}: {v}")
