# scripts/LLM/forms_llm/planner/loan_assistant_demo.py

from __future__ import annotations

import os
import sys
import json
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

# 1) Planner
from scripts.LLM.forms_llm.planner.query_planner_llm import plan_user_query_with_llm

# 2) Executor
from scripts.LLM.forms_llm.planner.plan_executor import execute_plan_on_session

# 3) Final answer LLM combiner
from scripts.LLM.forms_llm.planner.final_answer_llm import build_final_answer


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
    Save planner / executor / final-answer outputs per session so you can inspect later.

    Location (logical path):
      data/local_pipeline/sessions/<session_id>/rag_debug/<name>_<timestamp>.json

    Storage behaviour:
      - Always uses gcs_utils.write_json(), so in USE_GCS_OUTPUT=True debug files
        are written to GCS (and optionally mirrored locally if WRITE_LOCAL_COPY=True).
    """
    debug_dir = LIVE_SESSIONS_DIR / session_id / "rag_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)  # local dir (for easy browsing)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = debug_dir / f"{name}_{ts}.json"

    # GCS-aware write
    write_json(out_path, data)

    print(f"[debug] saved {name} -> {out_path}")


# ---------------------------------------------------------------------
# Main demo runner
# ---------------------------------------------------------------------
def run_loan_assistant_demo(user_query: str, session_id: str | None = None) -> str:
    """
    Full pipeline:
      user query -> planner -> executor -> final answer llm

    Args:
        user_query: natural-language question from the user
        session_id: optional session folder name (session_xxxx). If None,
                    the latest session is used.
    """
    if session_id is None:
        session_id = _get_latest_session_id()

    print(f"\nUsing session: {session_id}")
    print("USER QUERY:", user_query)

    # -----------------------------
    # 1) Planner LLM
    # -----------------------------
    plan = plan_user_query_with_llm(user_query, default_language="en")
    plan_dict = plan.to_dict()

    print("\nGenerated plan JSON:\n")
    print(json.dumps(plan_dict, ensure_ascii=False, indent=2))

    _save_debug_json(session_id, "plan", plan_dict)

    # -----------------------------
    # 2) Execute plan on session
    # -----------------------------
    exec_results = execute_plan_on_session(
        plan=plan,
        session_id=session_id,
        top_k_local=8,
    )

    _save_debug_json(session_id, "exec_results", exec_results)

    # -----------------------------
    # 3) Final answer LLM
    # -----------------------------
    # decide output language:
    # if planner tasks contain any non-en language, use the first one
    out_lang = "en"
    for t in plan.tasks:
        if t.language and t.language.lower() not in {"en", "english"}:
            out_lang = t.language
            break

    final_text = build_final_answer(
        user_query=user_query,
        plan=plan,
        doc_answers=exec_results.get("doc_answers", {}),
        finance_results=exec_results.get("finance_results", {}),
        output_language=out_lang,
    )

    print("\n=========== FINAL ANSWER ===========\n")
    print(final_text)
    print("\n====================================\n")

    _save_debug_json(session_id, "final_answer", {"text": final_text})

    return final_text


# ---------------------------------------------------------------------
# Quick interactive test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        "Explain my loan agreement and tell me the EMI if I borrow 10000 at 8% for 36 months.",
        "Summarise this contract and highlight any penalties for late payment.",
        "Explain this agreement in Telugu and also calculate EMI for 7000 at 9% for 24 months.",
        "What is the EMI from this agreement?",  # should return needs_input if numbers missing
    ]

    for q in tests:
        print("\n\n####################################")
        run_loan_assistant_demo(q)
