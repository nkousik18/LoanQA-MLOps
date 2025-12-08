# scripts/LLM/forms_llm/ui/streamlit_app.py
from __future__ import annotations

import os
import sys
import json
import tempfile
import importlib.util
from pathlib import Path
from typing import Callable

import streamlit as st

# ---------------------------------------------------------
# ✅ Setup project root on sys.path
# ---------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# Central config + GCS helpers
from scripts.aws_extraction_scripts.config import (
    LIVE_SESSIONS_DIR,
    GCS_BUCKET,
    USE_GCS_OUTPUT,
    USER_UPLOADS_DIR,
    to_gcs_key,
)

# ---------------------------------------------------------
# ✅ Load loan assistant runner
# ---------------------------------------------------------
def _load_loan_assistant_runner() -> Callable[..., str]:
    """Scan planner directory for loan assistant function."""
    planner_dir = PROJECT_ROOT / "scripts" / "LLM" / "forms_llm" / "planner"

    for file_path in planner_dir.glob("*.py"):
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        if (
            "run_loan_assistant_demo" in text
            or "run_loan_assistant" in text
            or "answer_query" in text
        ):
            spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
            mod = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(mod)

            for fn_name in [
                "run_loan_assistant_demo",
                "run_loan_assistant",
                "answer_query",
                "main",
            ]:
                if hasattr(mod, fn_name):
                    return getattr(mod, fn_name)

    raise ImportError("Could not find loan assistant runner in planner/*.py")


run_loan_assistant_demo = _load_loan_assistant_runner()

# =========================================================
# GCP helpers
# =========================================================
try:
    from google.cloud import storage
except ImportError:
    storage = None


def _get_gcs_bucket_name() -> str:
    """GCS bucket where PDFs live. Uses central config."""
    if not GCS_BUCKET:
        raise RuntimeError("GCS bucket not configured in config.py")
    return GCS_BUCKET


def _upload_pdf_to_gcs(local_pdf_path: str, object_name: str) -> None:
    """
    Upload local PDF to GCS.
    
    Args:
        local_pdf_path: Local temp file path
        object_name: GCS object key (e.g., "data/user_uploads/file.pdf")
    """
    if storage is None:
        raise RuntimeError(
            "google-cloud-storage not installed. Run: pip install google-cloud-storage"
        )

    bucket_name = _get_gcs_bucket_name()
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    blob.upload_from_filename(local_pdf_path, content_type="application/pdf")
    print(f"[GCS] uploaded {local_pdf_path} -> gs://{bucket_name}/{object_name}")


# =========================================================
# Single-PDF runner
# =========================================================
from scripts.aws_extraction_scripts.sync_gcs_to_s3 import sync_user_uploads
from scripts.aws_extraction_scripts.single_pdf_pipeline import process_single_pdf_session


def RUN_SINGLE_PDF(local_pdf_path: str) -> str:
    """
    End-to-end pipeline:
    
    1. Upload PDF to GCS at data/user_uploads/tmpXXX.pdf
    2. Sync from GCS data/user_uploads/ to S3 user_uploads/
    3. Run Textract + segmentation + RAG
    4. Return session_id
    """
    filename = Path(local_pdf_path).name
    
    # FIXED: Upload to data/user_uploads/ instead of user_uploads/
    gcs_upload_prefix = to_gcs_key(USER_UPLOADS_DIR)
    if not gcs_upload_prefix.endswith('/'):
        gcs_upload_prefix += '/'
    
    object_name = f"{gcs_upload_prefix}{filename}"  # e.g., "data/user_uploads/tmpXXX.pdf"

    # 1) Upload to GCS
    _upload_pdf_to_gcs(local_pdf_path, object_name)

    # 2) Sync from GCS data/user_uploads/ to S3 user_uploads/
    sync_user_uploads()

    # 3) The S3 key after sync will be: user_uploads/tmpXXX.pdf
    s3_key = f"user_uploads/{filename}"
    
    # 4) Run pipeline
    info = process_single_pdf_session(s3_key)

    return info["session_id"]


# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
st.set_page_config(page_title="Doc-Understand | Loan Assistant", layout="wide")

st.title("Doc-Understand – Loan/Contract Assistant")
st.caption("Upload a PDF → run pipeline → ask questions in one chat box.")

# Sidebar
with st.sidebar:
    st.header("Session")
    if st.button("Reset session / upload new PDF"):
        st.session_state.pop("session_id", None)
        st.session_state.pop("pdf_name", None)
        st.session_state.pop("processed", None)
        st.session_state.pop("chat", None)
        st.rerun()

    st.markdown("---")
    st.header("Debug")
    show_plan = st.checkbox("Show generated plan JSON", value=False)
    show_exec = st.checkbox("Show exec_results JSON", value=False)

# Initialize session state
if "processed" not in st.session_state:
    st.session_state.processed = False
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "chat" not in st.session_state:
    st.session_state.chat = []

# Upload PDF
uploaded = st.file_uploader("Upload your loan/contract PDF", type=["pdf"])

col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.subheader("Step 1 — Process PDF")

    if uploaded is None:
        st.info("Upload a PDF to start.")
    else:
        st.write(f"**File:** {uploaded.name}")

        if st.button("Run pipeline on this PDF", type="primary"):
            data_dir = PROJECT_ROOT / "data"
            data_dir.mkdir(parents=True, exist_ok=True)

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".pdf", dir=str(data_dir)
            ) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            with st.spinner(
                "Uploading to GCS → syncing to S3 → Textract → segmentation → building RAG..."
            ):
                try:
                    session_id = RUN_SINGLE_PDF(tmp_path)
                    st.session_state.session_id = session_id
                    st.session_state.pdf_name = uploaded.name
                    st.session_state.processed = True
                    st.success(f"✅ Pipeline complete! Session: **{session_id}**")
                except Exception as e:
                    st.session_state.processed = False
                    st.session_state.session_id = None
                    st.error(f"Pipeline failed:\n\n{e}")

            # Cleanup
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    if st.session_state.processed and st.session_state.session_id:
        st.markdown("---")
        st.success(f"Active session: **{st.session_state.session_id}**")
        st.caption(f"PDF: {st.session_state.get('pdf_name', 'unknown')}")

# Query box + chat
with col2:
    st.subheader("Step 2 — Ask Questions")

    if not st.session_state.processed or not st.session_state.session_id:
        st.warning("Process a PDF first. Then ask questions here.")
    else:
        # Chat history
        for msg in st.session_state.chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_q = st.chat_input("Ask about your loan/contract...")

        if user_q:
            st.session_state.chat.append({"role": "user", "content": user_q})
            with st.chat_message("user"):
                st.markdown(user_q)

            with st.chat_message("assistant"):
                with st.spinner("Planning → retrieving → computing → answering..."):
                    try:
                        final_text = run_loan_assistant_demo(
                            user_query=user_q,
                            session_id=st.session_state.session_id,
                        )
                        st.markdown(final_text)
                        st.session_state.chat.append(
                            {"role": "assistant", "content": final_text}
                        )
                    except Exception as e:
                        err = f"Error: {e}"
                        st.error(err)
                        st.session_state.chat.append(
                            {"role": "assistant", "content": err}
                        )

            # Debug views (GCS-aware)
            if show_plan or show_exec:
                rag_debug_dir = LIVE_SESSIONS_DIR / st.session_state.session_id / "rag_debug"

                def _load_latest_debug_json(prefix: str):
                    # Try local first
                    if rag_debug_dir.exists():
                        local_files = sorted(
                            rag_debug_dir.glob(f"{prefix}*.json"),
                            key=lambda p: p.name,
                            reverse=True,
                        )
                        if local_files:
                            try:
                                return json.loads(local_files[0].read_text(encoding="utf-8"))
                            except Exception:
                                pass

                    # Try GCS
                    if USE_GCS_OUTPUT and storage is not None:
                        client = storage.Client()
                        bucket = client.bucket(GCS_BUCKET)
                        dir_prefix = to_gcs_key(rag_debug_dir)
                        if not dir_prefix.endswith("/"):
                            dir_prefix += "/"
                        blobs = list(client.list_blobs(GCS_BUCKET, prefix=dir_prefix + prefix))
                        if blobs:
                            latest_blob = max(blobs, key=lambda b: b.name)
                            try:
                                return json.loads(latest_blob.download_as_text(encoding="utf-8"))
                            except Exception:
                                return None
                    return None

                if show_plan:
                    plan_json = _load_latest_debug_json("plan_")
                    if plan_json:
                        st.markdown("### Debug: Latest Plan JSON")
                        st.json(plan_json)
                    else:
                        st.info("No plan JSON found.")

                if show_exec:
                    exec_json = _load_latest_debug_json("exec_results_")
                    if exec_json:
                        st.markdown("### Debug: Latest exec_results JSON")
                        st.json(exec_json)
                    else:
                        st.info("No exec_results JSON found.")

# Footer
st.markdown("---")
st.caption("Doc-Understand | Planner + Span-RAG + Finance tools + Final Composer")