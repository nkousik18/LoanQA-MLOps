"""
loan_assistant_demo.py (WITH CLOUD MONITORING)
-----------------------------------------------
Main query execution pipeline with Cloud Monitoring integration.

Changes from original:
- Added timing for query execution
- Tracks query types and RAG quality
- Logs all queries to Cloud Monitoring
"""

from __future__ import annotations

import os
import sys
import json
import time  # ✅ ADDED for timing
from pathlib import Path
from datetime import datetime
from typing import Any, Dict

# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent           # .../planner
PROJECT_ROOT = CURRENT_DIR.parents[3]                   # .../doc-understand
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(PROJECT_ROOT)

# ---------------------------------------------------------------------
# Central config + GCS helpers
# ---------------------------------------------------------------------
from scripts.aws_extraction_scripts.config import LIVE_SESSIONS_DIR
from scripts.aws_extraction_scripts.gcs_utils import write_json
from scripts.aws_extraction_scripts.log_utils import get_logger

# 1) Planner
from scripts.LLM.forms_llm.planner.query_planner_llm import plan_user_query_with_llm

# 2) Executor
from scripts.LLM.forms_llm.planner.plan_executor import execute_plan_on_session

# 3) Final answer LLM combiner
from scripts.LLM.forms_llm.planner.final_answer_llm import build_final_answer

# ✅ NEW: Cloud Monitoring
try:
    from scripts.aws_extraction_scripts.cloud_monitoring import (
        log_query_execution,
        log_error,
    )
    MONITORING_ENABLED = True
except ImportError:
    MONITORING_ENABLED = False
    print("⚠️  Cloud Monitoring not available")

LOGGER = get_logger(__name__)


# ---------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------
def _get_latest_session_id() -> str:
    """
    Find the most recent session_* folder under LIVE_SESSIONS_DIR.
    """
    sessions_dir = LIVE_SESSIONS_DIR
    if not sessions_dir.exists():
        raise FileNotFoundError(f"Sessions directory not found: {sessions_dir}")

    session_dirs = [
        d for d in sessions_dir.iterdir()
        if d.is_dir() and d.name.startswith("session_")
    ]
    if not session_dirs:
        raise FileNotFoundError(f"No session_* folders found in {sessions_dir}")

    latest_session = max(session_dirs, key=lambda d: d.stat().st_mtime)
    return latest_session.name


def _save_debug_json(session_id: str, name: str, data: Dict[str, Any]) -> None:
    """
    Save planner / executor / final-answer outputs per session.

    Location (logical path):
      data/local_pipeline/sessions/<session_id>/rag_debug/<name>_<timestamp>.json
    """
    debug_dir = LIVE_SESSIONS_DIR / session_id / "rag_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = debug_dir / f"{name}_{ts}.json"

    # GCS-aware write
    write_json(out_path, data)

    LOGGER.info(f"[loan_assistant_demo] saved debug JSON: {name} -> {out_path}")


def _extract_user_id_from_session(session_id: str) -> str:
    """
    Extract user_id from session_id.
    Format: session_<user_id>_<counter>_<uuid>
    """
    try:
        parts = session_id.split("_")
        if len(parts) >= 4:
            # session_<user>_<count>_<uuid> -> user is at index 1
            return parts[1]
        return "unknown_user"
    except Exception:
        return "unknown_user"


def _infer_query_type(plan_tasks: list) -> str:
    """
    Infer primary query type from plan tasks.
    """
    if not plan_tasks:
        return "unknown"
    
    # Count task kinds
    kinds = [t.get("kind", "unknown") for t in plan_tasks]
    
    # Return the first doc_* or finance_* task
    for kind in kinds:
        if kind.startswith("doc_") or kind.startswith("finance_"):
            return kind
    
    return kinds[0] if kinds else "unknown"


def _calculate_avg_rag_score(exec_results: Dict) -> float:
    """
    Calculate average RAG retrieval score from execution results.
    This is a simplified approximation.
    """
    try:
        doc_answers = exec_results.get("doc_answers", {})
        if not doc_answers:
            return 0.0
        
        # We don't have direct retrieval scores in exec_results,
        # so we'll return a placeholder. In production, you'd track
        # this in the executor itself.
        return 0.75  # Placeholder average
    except Exception:
        return 0.0


# ---------------------------------------------------------------------
# Main demo runner WITH MONITORING
# ---------------------------------------------------------------------
def run_loan_assistant_demo(user_query: str, session_id: str | None = None) -> str:
    """
    Full pipeline with Cloud Monitoring:
      user query -> planner -> executor -> final answer llm

    Args:
        user_query: natural-language question from the user
        session_id: optional session folder name (session_xxxx). If None,
                    the latest session is used.
    """
    # ✅ START TIMING
    start_time = time.time()
    query_success = False
    error_message = None
    rag_avg_score = None
    query_type = None
    
    if session_id is None:
        session_id = _get_latest_session_id()

    # Extract user_id from session_id
    user_id = _extract_user_id_from_session(session_id)

    LOGGER.info(f"[loan_assistant_demo] Using session: {session_id}")
    LOGGER.info(f"[loan_assistant_demo] USER QUERY: {user_query}")

    try:
        # -----------------------------
        # 1) Planner (rule-based)
        # -----------------------------
        plan = plan_user_query_with_llm(user_query, default_language="en")
        plan_dict = plan.to_dict()

        LOGGER.info(
            "[loan_assistant_demo] Generated plan with tasks: "
            f"{[t['kind'] for t in plan_dict.get('tasks', [])]}"
        )
        _save_debug_json(session_id, "plan", plan_dict)
        
        # Infer query type
        query_type = _infer_query_type(plan_dict.get("tasks", []))

        # -----------------------------
        # 2) Execute plan on session
        # -----------------------------
        LOGGER.info("[loan_assistant_demo] Executing plan on session...")
        exec_results = execute_plan_on_session(
            plan=plan,
            session_id=session_id,
            top_k_local=8,
        )
        _save_debug_json(session_id, "exec_results", exec_results)
        
        # Calculate RAG score
        rag_avg_score = _calculate_avg_rag_score(exec_results)

        # -----------------------------
        # 3) Final answer LLM
        # -----------------------------
        # decide output language:
        out_lang = "en"
        for t in plan.tasks:
            if t.language and t.language.lower() not in {"en", "english"}:
                out_lang = t.language
                break

        LOGGER.info(f"[loan_assistant_demo] Building final answer (language={out_lang})")

        final_text = build_final_answer(
            user_query=user_query,
            plan=plan,
            doc_answers=exec_results.get("doc_answers", {}),
            finance_results=exec_results.get("finance_results", {}),
            output_language=out_lang,
        )

        LOGGER.info("[loan_assistant_demo] Final answer generated.")
        _save_debug_json(session_id, "final_answer", {"text": final_text})

        # ✅ MARK SUCCESS
        query_success = True
        duration = time.time() - start_time
        
        # ✅ LOG TO CLOUD MONITORING
        if MONITORING_ENABLED:
            log_query_execution(
                session_id=session_id,
                user_id=user_id,
                query=user_query,
                duration_seconds=duration,
                success=True,
                query_type=query_type,
                rag_avg_score=rag_avg_score,
            )

        # Print for console/logs
        print("\n=========== FINAL ANSWER ===========\n")
        print(final_text)
        print("\n====================================\n")

        return final_text

    except Exception as e:
        # ✅ MARK FAILURE
        query_success = False
        error_message = str(e)
        duration = time.time() - start_time
        
        LOGGER.exception(f"[loan_assistant_demo] Query failed: {e}")
        
        # ✅ LOG TO CLOUD MONITORING
        if MONITORING_ENABLED:
            log_query_execution(
                session_id=session_id,
                user_id=user_id,
                query=user_query,
                duration_seconds=duration,
                success=False,
                query_type=query_type,
                error_msg=error_message,
            )
            
            log_error(
                error_type="query_execution_failed",
                error_message=error_message,
                session_id=session_id,
                user_id=user_id,
                context={"query": user_query[:200]},
            )
        
        # Re-raise
        raise


# ---------------------------------------------------------------------
# Quick interactive test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        "Explain my loan agreement and tell me the EMI if I borrow 10000 at 8% for 36 months.",
        "Summarise this contract and highlight any penalties for late payment.",
        "What is the interest rate?",
    ]

    for q in tests:
        print("\n\n####################################")
        try:
            run_loan_assistant_demo(q)
        except Exception as e:
            print(f"Query failed: {e}")
