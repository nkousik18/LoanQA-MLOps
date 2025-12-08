"""
single_pdf_pipeline.py
----------------------
Single-PDF pipeline for live user uploads.

Flow:
  (optional) GCP → S3 sync for user_uploads/
  1. Textract OCR  -> raw JSON  + raw text
  2. Segmentation  -> line-level spans
  3. Session RAG   -> build chunks/blocks/embeddings under session/rag/

Outputs are stored under a per-session folder:
  data/local_pipeline/sessions/<session_id>/{raw, raw_text, segmented, rag}
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
from scripts.aws_extraction_scripts.log_utils import get_logger
from scripts.aws_extraction_scripts.config import LIVE_SESSIONS_DIR

# ✅ Session RAG builder now uses SEGMENTED text
from scripts.LLM.forms_llm.rag_reasoning_scripts.session_rag_builder import (
    build_session_rag_from_segmented,
)

# 🔁 Optional GCP → S3 sync for user_uploads/
try:
    from scripts.aws_extraction_scripts.sync_gcs_to_s3 import sync_user_uploads
except Exception:  # if library/env not available, we just skip GCP sync
    sync_user_uploads = None

logger = get_logger("single_pdf_pipeline")

# Root for all live sessions (use central config constant)
SESSIONS_ROOT = LIVE_SESSIONS_DIR


# ---------------------------------------------------------------------
# 📁 Session directory helper
# ---------------------------------------------------------------------
def make_session_dirs(session_id: str) -> dict:
    """
    Create per-session directories and return all paths.
    These are local paths; the actual data can also be mirrored to GCS
    because run_textract_for_pdf_to_dirs + segment_textract_json_to_dir
    use gcs_utils under the hood.
    """
    session_root = SESSIONS_ROOT / session_id
    raw_dir = session_root / "raw"
    raw_text_dir = session_root / "raw_text"
    segmented_dir = session_root / "segmented"
    rag_dir = session_root / "rag"  # ✅ executor + RAG expect this

    for d in (raw_dir, raw_text_dir, segmented_dir, rag_dir):
        d.mkdir(parents=True, exist_ok=True)

    return {
        "session_id": session_id,
        "session_root": str(session_root),
        "raw_dir": raw_dir,
        "raw_text_dir": raw_text_dir,
        "segmented_dir": segmented_dir,
        "rag_dir": rag_dir,
    }


# ---------------------------------------------------------------------
# 🚀 Main single-PDF pipeline
# ---------------------------------------------------------------------
def process_single_pdf_session(
    pdf_s3_key: str,
    session_id: str | None = None,
) -> dict:
    """
    Run the full OCR → segmentation → session RAG pipeline for ONE PDF.

    Args:
        pdf_s3_key: S3 key, e.g. "user_uploads/loan1.pdf"
        session_id: optional fixed session id; if None, a random one is created.

    Returns:
        dict with paths for all outputs (raw_json, raw_text, segmented, rag).
    """
    # STEP 0: Optional GCP → S3 sync (user_uploads/)
    if sync_user_uploads is not None:
        try:
            logger.info("🔁 GCP→S3 sync: syncing user_uploads/ before Textract...")
            sync_user_uploads()
        except Exception as e:
            # Don't kill the pipeline – just log and fall back to existing S3 objects
            logger.warning(
                "⚠️ GCP→S3 sync failed, continuing with existing S3 objects only: %s",
                e,
            )
    else:
        logger.info(
            "ℹ️ GCP sync not enabled (sync_user_uploads import failed) – "
            "using existing S3 user_uploads/ only."
        )

    # Normal single-PDF pipeline from here ↓
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

    # --- Step 3: Build Session RAG (using SEGMENTED spans) ---
    rag_info = build_session_rag_from_segmented(str(segmented_path))
    # rag_info already contains rag_dir, chunks, blocks, embeddings paths

    info = {
        "session_id": session_id,
        "session_root": dirs["session_root"],
        "raw_json": raw_json_path,
        "raw_text": raw_text_path,
        "segmented": segmented_path,
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
    # Make sure this key exists:
    # - in GCP user_uploads/ (if you want sync)
    # - or directly in S3 user_uploads/ (if you skip GCP)
    test_pdf_key = "user_uploads/Generative Project.pdf"
    info = process_single_pdf_session(test_pdf_key)
    print("\n=== Single PDF Session Info ===")
    for k, v in info.items():
        print(f"{k}: {v}")
