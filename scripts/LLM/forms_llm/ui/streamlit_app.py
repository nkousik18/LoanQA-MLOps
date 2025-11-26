# scripts/ui/streamlit_app.py
from __future__ import annotations

import os
import sys
import tempfile
import uuid
import importlib.util
from pathlib import Path
from typing import Callable

import streamlit as st


# ---------------------------------------------------------
# ✅ Setup project root on sys.path
# file: doc-understand/scripts/LLM/forms_llm/ui/streamlit_app.py
# CURRENT_DIR = .../scripts/LLM/forms_llm/ui
# PROJECT_ROOT must be .../doc-understand  => parents[3]
# ---------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[3]   # 👈 change 1 → 3
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


# ---------------------------------------------------------
# ✅ Load loan assistant runner (robust to filename changes)
# ---------------------------------------------------------
def _load_loan_assistant_runner() -> Callable[..., str]:
    """
    Scan scripts/LLM/forms_llm/planner/ for a function that can answer user queries.
    Prefers run_loan_assistant_demo if present.
    """
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

    raise ImportError(
        "Could not find a loan assistant runner in scripts/LLM/forms_llm/planner/*.py.\n"
        "Make sure your planner file is saved and contains "
        "run_loan_assistant_demo (or run_loan_assistant / answer_query)."
    )


run_loan_assistant_demo = _load_loan_assistant_runner()


# ---------------------------------------------------------
# ✅ REAL single-PDF pipeline runner for YOUR AWS pipeline
# - local PDF -> upload to S3 -> process_single_pdf_session(s3_key)
# ---------------------------------------------------------
def _get_bucket_name() -> str:
    """
    Priority:
      1) env var (if set)
      2) scripts.aws_extraction_scripts.config.BUCKET (your config default)
    """
    # 1) env vars
    for k in ["S3_BUCKET", "AWS_TEXTRACT_BUCKET", "TEXTRACT_BUCKET", "AWS_BUCKET"]:
        v = os.getenv(k)
        if v:
            return v

    # 2) fallback to your config.py (you defined BUCKET there)
    try:
        from scripts.aws_extraction_scripts import config as cfg
        if hasattr(cfg, "BUCKET") and isinstance(cfg.BUCKET, str) and cfg.BUCKET.strip():
            return cfg.BUCKET.strip()
    except Exception:
        pass

    raise RuntimeError(
        "S3 bucket not found. Set env var S3_BUCKET or define BUCKET in "
        "scripts/aws_extraction_scripts/config.py."
    )


def _get_region_name() -> str:
    """
    Optional: read region from config if present.
    """
    try:
        from scripts.aws_extraction_scripts import config as cfg
        if hasattr(cfg, "REGION") and isinstance(cfg.REGION, str) and cfg.REGION.strip():
            return cfg.REGION.strip()
    except Exception:
        pass
    return os.getenv("AWS_REGION", "us-east-1")


def _upload_pdf_to_s3(local_pdf_path: str) -> str:
    """
    Upload local PDF to S3 and return the S3 key.
    """
    try:
        import boto3
    except ImportError:
        raise RuntimeError("boto3 not installed. Run: pip install boto3")

    bucket = _get_bucket_name()
    region = _get_region_name()

    s3 = boto3.client("s3", region_name=region)

    filename = Path(local_pdf_path).name
    s3_key = f"user_uploads/{filename}"

    s3.upload_file(local_pdf_path, bucket, s3_key)
    return s3_key


def RUN_SINGLE_PDF(local_pdf_path: str) -> str:
    """
    End-to-end:
      local file -> S3 -> your pipeline -> session_id
    """
    # 1) upload to s3
    s3_key = _upload_pdf_to_s3(local_pdf_path)

    # 2) run your single pdf session pipeline
    from scripts.aws_extraction_scripts.single_pdf_pipeline import process_single_pdf_session
    info = process_single_pdf_session(s3_key)

    # 3) return session id
    return info["session_id"]


# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
st.set_page_config(page_title="Doc-Understand | Loan Assistant", layout="wide")

st.title("Doc-Understand – Loan/Contract Assistant")
st.caption("Upload a PDF → run pipeline → ask questions in one chat box.")


# Sidebar controls
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
    st.session_state.chat = []  # list of {"role": "user"/"assistant", "content": str}


# ---------------------------------------------------------
# 1) Upload PDF
# ---------------------------------------------------------
uploaded = st.file_uploader("Upload your loan/contract PDF", type=["pdf"])

col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.subheader("Step 1 — Process PDF")

    if uploaded is None:
        st.info("Upload a PDF to start.")
    else:
        st.write(f"**File:** {uploaded.name}")

        if st.button("Run pipeline on this PDF", type="primary"):
            # Save uploaded PDF to a temp file in your repo data folder
            data_dir = PROJECT_ROOT / "data"
            data_dir.mkdir(parents=True, exist_ok=True)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir=str(data_dir)) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            with st.spinner("Uploading to S3 → Textract → segmentation → normalization → layout..."):
                try:
                    session_id = RUN_SINGLE_PDF(tmp_path)
                    st.session_state.session_id = session_id
                    st.session_state.pdf_name = uploaded.name
                    st.session_state.processed = True
                    st.success(f"Pipeline complete. Session created: **{session_id}**")
                except Exception as e:
                    st.session_state.processed = False
                    st.session_state.session_id = None
                    st.error(f"Pipeline failed:\n\n{e}")

            # cleanup temp file
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    if st.session_state.processed and st.session_state.session_id:
        st.markdown("---")
        st.success(f"Active session: **{st.session_state.session_id}**")
        st.caption(f"PDF: {st.session_state.get('pdf_name', 'unknown')}")


# ---------------------------------------------------------
# 2) Query box + chat
# ---------------------------------------------------------
with col2:
    st.subheader("Step 2 — Ask Questions")

    if not st.session_state.processed or not st.session_state.session_id:
        st.warning("Process a PDF first. Then ask questions here.")
    else:
        # Show chat history
        for msg in st.session_state.chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_q = st.chat_input(
            "Type your question (e.g., explain my loan and compute EMI for 10000 at 8% for 36 months)"
        )

        if user_q:
            # Add user message
            st.session_state.chat.append({"role": "user", "content": user_q})
            with st.chat_message("user"):
                st.markdown(user_q)

            # Run full demo pipeline
            with st.chat_message("assistant"):
                with st.spinner("Planning → retrieving clauses → computing tools → writing final answer..."):
                    try:
                        final_text = run_loan_assistant_demo(
                            user_query=user_q,
                            session_id=st.session_state.session_id,
                        )
                        st.markdown(final_text)
                        st.session_state.chat.append({"role": "assistant", "content": final_text})
                    except Exception as e:
                        err = f"Error while answering:\n\n{e}"
                        st.error(err)
                        st.session_state.chat.append({"role": "assistant", "content": err})

            # Optional debug views
            if show_plan or show_exec:
                rag_debug_dir = (
                    PROJECT_ROOT
                    / "data"
                    / "local_pipeline"
                    / "sessions"
                    / st.session_state.session_id
                    / "rag_debug"
                )
                if rag_debug_dir.exists():
                    newest = sorted(
                        rag_debug_dir.glob("*.json"),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True
                    )

                    if show_plan:
                        plan_files = [p for p in newest if p.name.startswith("plan_")]
                        if plan_files:
                            st.markdown("### Debug: Latest Plan JSON")
                            st.json(plan_files[0].read_text(encoding="utf-8"))
                        else:
                            st.info("No plan JSON found yet.")

                    if show_exec:
                        exec_files = [p for p in newest if p.name.startswith("exec_results_")]
                        if exec_files:
                            st.markdown("### Debug: Latest exec_results JSON")
                            st.json(exec_files[0].read_text(encoding="utf-8"))
                        else:
                            st.info("No exec_results JSON found yet.")
                else:
                    st.info("No rag_debug folder found for this session.")


# Footer
st.markdown("---")
st.caption("Doc-Understand | Planner + Span-RAG + Finance tools + Final Composer")
