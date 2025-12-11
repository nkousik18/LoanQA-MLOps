"""
single_pdf_pipeline.py
----------------------
Single-PDF pipeline for live user uploads with user-aware session naming.

Flow:
  1. GCP → S3 sync for user_uploads/
  2. Textract OCR → raw JSON + raw text (PII masked)
  3. Segmentation → line-level spans  
  4. Session RAG → chunks/blocks/embeddings
  5. Metadata → session info with user tracking

Outputs stored per session (GCS + local):
  data/local_pipeline/sessions/session_<user>_<count>_<id>/{raw, segmented, rag, meta.json}
"""

from __future__ import annotations

import os
import sys
import json
from datetime import datetime
from pathlib import Path

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
from scripts.aws_extraction_scripts.run_textract import run_textract_for_pdf_to_dirs
from scripts.aws_extraction_scripts.segment_text import segment_textract_json_to_dir
from scripts.aws_extraction_scripts.log_utils import get_logger
from scripts.aws_extraction_scripts.tracker import track_task
from scripts.aws_extraction_scripts.config import LIVE_SESSIONS_DIR

from scripts.LLM.forms_llm.rag_reasoning_scripts.session_rag_builder import (
    build_session_rag_from_segmented,
)

# GCP → S3 sync
try:
    from scripts.aws_extraction_scripts.sync_gcs_to_s3 import sync_user_uploads
except Exception:
    sync_user_uploads = None

# Session naming helpers
from scripts.aws_extraction_scripts.session_naming import (
    clean_filename,
    get_next_upload_counter,
    make_session_id,
    write_session_metadata,
)

LOGGER = get_logger(__name__)
SESSIONS_ROOT = LIVE_SESSIONS_DIR


# ---------------------------------------------------------------------
# Create per-session directories
# ---------------------------------------------------------------------
def make_session_dirs(session_id: str) -> dict:
    """Create per-session directories (local for RAG processing)."""
    session_root = SESSIONS_ROOT / session_id

    raw_dir = session_root / "raw"
    raw_text_dir = session_root / "raw_text"
    segmented_dir = session_root / "segmented"
    rag_dir = session_root / "rag"

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
# Main pipeline with user-aware session naming
# ---------------------------------------------------------------------
def process_single_pdf_session(
    pdf_s3_key: str,
    user_id: str = "default_user",
    session_id: str | None = None,
) -> dict:
    """
    Process one PDF with user-aware session tracking.

    Args:
        pdf_s3_key: S3 key (already stripped of data/ prefix by streamlit_app.py)
                   e.g., "user_uploads/tmp12345.pdf"
        user_id: User identifier (default: "default_user")
        session_id: Optional fixed session ID

    Returns:
        dict with session info and file paths
    """
    # STEP 0: Sync from GCS to S3
    if sync_user_uploads:
        try:
            LOGGER.info("🔁 Syncing user_uploads/ from GCS → S3...")
            sync_user_uploads()
        except Exception as e:
            LOGGER.warning(f"⚠️ GCS sync failed: {e}")

    # ---------------------------
    # Assign structured session ID
    # ---------------------------
    filename = pdf_s3_key.split("/")[-1]
    cleaned_filename = clean_filename(filename)

    if session_id is None:
        upload_counter = get_next_upload_counter(user_id)
        session_id = make_session_id(user_id, upload_counter)
        LOGGER.info(f"📛 Created session ID: {session_id}")
    else:
        try:
            upload_counter = int(session_id.split("_")[2])
        except (IndexError, ValueError):
            upload_counter = 1

    dirs = make_session_dirs(session_id)

    # Early metadata
    session_meta = {
        "session_id": session_id,
        "user_id": user_id,
        "upload_counter": upload_counter,
        "original_filename": cleaned_filename,
        "s3_key": pdf_s3_key,
        "timestamp": datetime.utcnow().isoformat(),
    }
    write_session_metadata(Path(dirs["session_root"]), session_meta)

    # Tracking name for the whole pipeline
    pipeline_task_name = f"single_pdf_session_{session_id}"
    track_task(pipeline_task_name, "STARTED")

    try:
        # ---------------------------
        # Step 1: Textract OCR (PII masked)
        # ---------------------------
        LOGGER.info("📄 Step 1: Running Textract OCR with PII masking...")
        raw_json_path, raw_text_path = run_textract_for_pdf_to_dirs(
            pdf_s3_key,
            dirs["raw_dir"],
            dirs["raw_text_dir"],
        )
        if raw_json_path is None:
            raise RuntimeError(f"Textract failed for {pdf_s3_key}")

        # ---------------------------
        # Step 2: Segmentation
        # ---------------------------
        LOGGER.info("✂️ Step 2: Segmenting text...")
        segmented_path = segment_textract_json_to_dir(
            raw_json_path,
            dirs["segmented_dir"],
        )
        if segmented_path is None:
            raise RuntimeError(f"Segmentation failed for {raw_json_path}")

        # ---------------------------
        # Step 3: Build RAG
        # ---------------------------
        LOGGER.info("🔍 Step 3: Building RAG artifacts...")
        rag_info = build_session_rag_from_segmented(str(segmented_path))

        # Update metadata with RAG info
        session_meta.update(
            {
                "raw_json": raw_json_path,
                "raw_text": raw_text_path,
                "segmented": segmented_path,
                "rag_dir": rag_info.get("rag_dir"),
                "rag_chunks": rag_info.get("chunks_path"),
                "rag_blocks": rag_info.get("blocks_path"),
                "rag_embeddings": rag_info.get("embeddings_path"),
                "num_chunks": rag_info.get("num_chunks"),
                "num_blocks": rag_info.get("num_blocks"),
            }
        )

        # Save final metadata (GCS-aware)
        write_session_metadata(Path(dirs["session_root"]), session_meta)

        LOGGER.info("✅ Pipeline complete for session: %s", session_id)
        track_task(
            pipeline_task_name,
            "SUCCESS",
            details=f"Completed single PDF session for user={user_id}, session_id={session_id}",
        )
        return session_meta

    except Exception as e:
        LOGGER.exception(f"❌ Pipeline failed for session {session_id}: {e}")
        track_task(
            pipeline_task_name,
            "FAILED",
            error=str(e),
        )
        # Re-raise so Streamlit / caller can surface the error
        raise


# ---------------------------------------------------------------------
# Manual test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Test with user folder structure
    # Note: Streamlit will strip "data/" prefix before passing to this function
    # So we test with S3 key format (no "data/" prefix)
    test_key = "user_uploads/test_user/form_centre_showcase_personal_loan_form.pdf"
    result = process_single_pdf_session(test_key, user_id="test_user")
    print("\n=== Single PDF Session Info ===")
    print(json.dumps(result, indent=2))

