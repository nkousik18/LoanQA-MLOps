# scripts/LLM/forms_llm/ui/streamlit_app_monitored.py
"""
Streamlit App WITH Cloud Monitoring Integration
------------------------------------------------
This version uses the monitored pipeline files.

Run with:
    streamlit run scripts/LLM/forms_llm/ui/streamlit_app_monitored.py
"""

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

# Session naming
from scripts.aws_extraction_scripts.session_naming import clean_filename

# ---------------------------------------------------------
# ✅ Load loan assistant runner (MONITORED VERSION)
# ---------------------------------------------------------
def _load_loan_assistant_runner() -> Callable[..., str]:
    """Scan planner directory for loan assistant function."""
    planner_dir = PROJECT_ROOT / "scripts" / "LLM" / "forms_llm" / "planner"

    # Try monitored version first, then fall back to regular
    for filename in ["loan_assistant_demo_monitored.py", "loan_assistant_demo.py"]:
        file_path = planner_dir / filename
        if not file_path.exists():
            continue
        
        try:
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
                    print(f"✅ Loaded query handler from: {filename}")
                    return getattr(mod, fn_name)
        except Exception as e:
            print(f"⚠️  Failed to load {filename}: {e}")
            continue

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
    """GCS bucket name from config."""
    if not GCS_BUCKET:
        raise RuntimeError("GCS bucket not configured in config.py")
    return GCS_BUCKET


def _upload_pdf_to_gcs(local_pdf_path: str, object_name: str) -> None:
    """Upload local PDF to GCS."""
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
# Single-PDF runner with user folder support (MONITORED VERSION)
# =========================================================
from scripts.aws_extraction_scripts.sync_gcs_to_s3 import sync_user_uploads

# ✅ Import MONITORED version
try:
    from scripts.aws_extraction_scripts.single_pdf_pipeline_monitored import (
        process_single_pdf_session
    )
    print("✅ Using MONITORED single_pdf_pipeline")
except ImportError:
    from scripts.aws_extraction_scripts.single_pdf_pipeline import (
        process_single_pdf_session
    )
    print("⚠️  Using regular single_pdf_pipeline (no monitoring)")


def RUN_SINGLE_PDF(local_pdf_path: str, user_id: str = "default_user") -> str:
    """
    End-to-end pipeline with user-specific folder organization + MONITORING.
    
    Flow:
    1. Upload PDF to GCS: data/user_uploads/<user_id>/file.pdf
    2. Sync to S3: user_uploads/<user_id>/file.pdf
    3. Run Textract + segmentation + RAG (WITH MONITORING)
    4. Create session with user tracking
    
    Args:
        local_pdf_path: Local temp file path
        user_id: User identifier (default: "default_user")
    
    Returns:
        session_id (e.g., "session_john_001_abc123")
    """
    filename = Path(local_pdf_path).name
    
    # Get base upload prefix (data/user_uploads/)
    gcs_upload_prefix = to_gcs_key(USER_UPLOADS_DIR)
    if not gcs_upload_prefix.endswith('/'):
        gcs_upload_prefix += '/'
    
    # Add user folder: data/user_uploads/<user_id>/file.pdf
    object_name = f"{gcs_upload_prefix}{user_id}/{filename}"
    
    print(f"📤 Uploading to GCS: {object_name}")

    # 1) Upload to GCS with user folder
    _upload_pdf_to_gcs(local_pdf_path, object_name)

    # 2) Sync from GCS data/user_uploads/ to S3 user_uploads/
    sync_user_uploads()

    # 3) The S3 key after sync: user_uploads/<user_id>/file.pdf
    s3_key = f"user_uploads/{user_id}/{filename}"
    
    print(f"🔄 S3 key for Textract: {s3_key}")
    
    # 4) Run pipeline with user_id (MONITORED VERSION)
    result = process_single_pdf_session(s3_key, user_id=user_id)

    return result["session_id"]


# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
st.set_page_config(page_title="Doc-Understand | Loan Assistant (Monitored)", layout="wide")

st.title("📊 Doc-Understand – Loan/Contract Assistant")
st.caption("✅ **WITH CLOUD MONITORING** - Upload a PDF → run pipeline → ask questions")

# Sidebar
with st.sidebar:
    st.header("Session")
    
    # User ID input
    user_id_input = st.text_input(
        "User ID (optional)", 
        value="default_user",
        help="Enter your user ID or use 'default_user'"
    )
    
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
    
    st.markdown("---")
    st.header("📊 Monitoring")
    st.success("✅ Cloud Monitoring Enabled")
    st.caption("Metrics are being sent to Google Cloud Monitoring")
    
    if st.button("🔗 Open Cloud Console"):
        st.markdown(
            "[Open Monitoring Dashboard](https://console.cloud.google.com/monitoring?project=doc-understand)",
            unsafe_allow_html=True
        )

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
        st.write(f"**User:** {user_id_input}")

        if st.button("Run pipeline on this PDF", type="primary"):
            data_dir = PROJECT_ROOT / "data"
            data_dir.mkdir(parents=True, exist_ok=True)

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".pdf", dir=str(data_dir)
            ) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            with st.spinner(
                "📊 Processing with monitoring enabled... "
                "Uploading to GCS → syncing to S3 → Textract → segmentation → building RAG..."
            ):
                try:
                    # Pass user_id to RUN_SINGLE_PDF (MONITORED VERSION)
                    session_id = RUN_SINGLE_PDF(tmp_path, user_id=user_id_input)
                    st.session_state.session_id = session_id
                    st.session_state.pdf_name = uploaded.name
                    st.session_state.user_id = user_id_input
                    st.session_state.processed = True
                    st.success(f"✅ Pipeline complete! Session: **{session_id}**")
                    st.info("📊 Metrics sent to Cloud Monitoring")
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
        st.caption(f"User: {st.session_state.get('user_id', 'unknown')}")

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
                with st.spinner("📊 Planning → retrieving → computing → answering (monitoring enabled)..."):
                    try:
                        # Uses MONITORED version
                        final_text = run_loan_assistant_demo(
                            user_query=user_q,
                            session_id=st.session_state.session_id,
                        )
                        st.markdown(final_text)
                        st.session_state.chat.append(
                            {"role": "assistant", "content": final_text}
                        )
                        st.caption("📊 Query metrics sent to Cloud Monitoring")
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
st.caption("Doc-Understand | 📊 WITH CLOUD MONITORING | Planner + Span-RAG + Finance tools + Final Composer")
st.caption("Metrics tracked: PDF processing time, query response time, RAG quality, success rates, errors")