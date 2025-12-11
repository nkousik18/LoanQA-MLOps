# scripts/LLM/forms_llm/planner/final_answer_llm.py

from __future__ import annotations

import os
import sys
import json
from functools import lru_cache
from typing import Dict, Any

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.LLM.forms_llm.planner.plan_schema import PlannerPlan
from scripts.LLM.forms_llm.llm_clients.groq_client import call_groq_chat

# ✅ prompts live here: scripts/LLM/prompts_form/
PROMPTS_DIR = os.path.join(PROJECT_ROOT, "scripts", "LLM", "prompts_form")

# Reuse central logger infra
from scripts.aws_extraction_scripts.log_utils import get_logger

LOGGER = get_logger(__name__)


@lru_cache(maxsize=4)
def _load_prompt_file(name: str) -> str:
    """
    Load a prompt template from scripts/LLM/prompts_form/<name>.
    Cached so we don't keep hitting disk.
    """
    path = os.path.join(PROMPTS_DIR, name)
    if not os.path.exists(path):
        LOGGER.error(f"[final_answer_llm] Prompt file not found: {path}")
        raise FileNotFoundError(f"Prompt file not found: {path}")
    LOGGER.debug(f"[final_answer_llm] Loading prompt file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _language_instruction_for_final(language: str) -> str:
    lang = (language or "").lower()
    if lang in {"", "en", "english"}:
        return "Respond in clear, concise English."
    if lang in {"te", "telugu"}:
        return (
            "Respond primarily in natural Telugu using Telugu script. "
            "Keep all numbers, dates, and amounts exactly as given in the task results."
        )
    # generic fallback
    return (
        f"Respond primarily in {language}. "
        "Preserve all numbers, dates, interest rates, and amounts exactly as given."
    )


def _fmt(val: Any, default: str = "not available") -> str:
    if val is None:
        return default
    try:
        # keep ints clean, floats readable
        if isinstance(val, (int, float)):
            return f"{val:.6g}"
        return str(val)
    except Exception:
        return default


def build_final_answer(
    user_query: str,
    plan: PlannerPlan,
    doc_answers: Dict[str, Any],
    finance_results: Dict[str, Any],
    output_language: str = "en",
) -> str:
    """
    Combine task-level outputs (doc + finance) into ONE user-facing answer
    using the LLM.

    We DO NOT re-read the PDF here. We only use:
      - plan JSON
      - doc_answers texts
      - numeric tool results
    """
    LOGGER.info(
        "[final_answer_llm] Building final answer "
        f"(language={output_language}, "
        f"doc_tasks={len(doc_answers or {})}, "
        f"finance_tasks={len(finance_results or {})})"
    )

    # 🔹 Load system prompt from external file + language hint
    base_prompt = _load_prompt_file("final_answer_prompt.txt")
    system_prompt = base_prompt + "\n\n" + _language_instruction_for_final(output_language)

    # -----------------------------
    # Clean doc answers formatting (NO bracket labels)
    # -----------------------------
    doc_results_text_parts = []
    for task_id, ans in (doc_answers or {}).items():
        doc_results_text_parts.append(
            f"Doc task {task_id} ({ans.get('kind')} | language={ans.get('language')}):\n"
            f"{(ans.get('answer') or '').strip()}\n"
        )
    doc_results_text = (
        "\n\n".join(doc_results_text_parts) or "No document tasks were executed."
    )

    # -----------------------------
    # Robust finance formatting (NO bracket labels)
    # -----------------------------
    fin_results_text_parts = []
    for task_id, res in (finance_results or {}).items():
        kind = res.get("kind")
        status = res.get("status", "ok")

        # If this finance task is pending or unsupported, surface that cleanly
        if status != "ok":
            fin_results_text_parts.append(
                f"Finance task {task_id} ({kind}) status={status}:\n"
                f"Note: {res.get('note', 'Finance tool could not run yet.')}\n"
                + (
                    f"Missing fields: {', '.join(res.get('missing_fields', []))}\n"
                    if res.get("missing_fields")
                    else ""
                )
                + (
                    f"Parsed payload: {json.dumps(res.get('parsed_payload', {}), ensure_ascii=False)}\n"
                    if res.get("parsed_payload")
                    else ""
                )
            )
            continue

        currency = res.get("currency", "")
        payment_amount = res.get("payment_amount")
        total_paid = res.get("total_paid")
        total_interest = res.get("total_interest")
        annual_rate_percent = res.get("annual_rate_percent")
        tenure_months = res.get("tenure_months")

        principal_approx = None
        try:
            if total_paid is not None and total_interest is not None:
                principal_approx = float(total_paid) - float(total_interest)
        except Exception:
            principal_approx = None

        fin_results_text_parts.append(
            f"Finance task {task_id} ({kind}) status=ok:\n"
            f"Principal (approx): {_fmt(principal_approx)} {currency}\n"
            f"Annual rate: {_fmt(annual_rate_percent)}%\n"
            f"Tenure: {_fmt(tenure_months)} months\n"
            f"EMI/payment amount: {_fmt(payment_amount)} {currency}\n"
            f"Total paid: {_fmt(total_paid)} {currency}\n"
            f"Total interest: {_fmt(total_interest)} {currency}\n"
        )

    fin_results_text = (
        "\n\n".join(fin_results_text_parts) or "No finance tools were executed."
    )

    # Plan JSON pretty-print (helps the LLM see structure)
    plan_json = json.dumps(plan.to_dict(), ensure_ascii=False, indent=2)

    user_prompt = (
        f"User query:\n{user_query}\n\n"
        "TASK PLAN (JSON):\n"
        f"{plan_json}\n\n"
        "DOCUMENT TASK RESULTS:\n"
        f"{doc_results_text}\n\n"
        "FINANCE TOOL RESULTS:\n"
        f"{fin_results_text}\n\n"
        "Final answer instructions from the planner:\n"
        f"{plan.final_answer_instructions}\n\n"
        "Now, following the plan and the instructions above, write ONE final answer "
        "for the user. Do not repeat the entire task results verbatim; synthesize them "
        "into a clean response."
    )

    LOGGER.debug(
        "[final_answer_llm] Calling Groq LLM for final answer "
        f"(user_query_len={len(user_query)}, "
        f"doc_text_len={len(doc_results_text)}, "
        f"fin_text_len={len(fin_results_text)})"
    )
    final_text = call_groq_chat(system_prompt, user_prompt)

    LOGGER.info(
        "[final_answer_llm] Final answer generated "
        f"(length={len(final_text)})"
    )
    return final_text


# -----------------------------
# Manual test
# -----------------------------
if __name__ == "__main__":
    from pathlib import Path
    from scripts.LLM.forms_llm.planner.plan_schema import PlannerPlan, PlannedTask
    from scripts.LLM.forms_llm.planner.plan_executor import execute_plan_on_session

    # Pick latest session
    sessions_dir = Path(PROJECT_ROOT) / "data" / "local_pipeline" / "sessions"
    session_dirs = [
        d for d in sessions_dir.iterdir()
        if d.is_dir() and d.name.startswith("session_")
    ]
    if not session_dirs:
        print(f"No session_* folders found in {sessions_dir}")
        sys.exit(1)

    latest_session = max(session_dirs, key=lambda d: d.stat().st_mtime)
    session_id = latest_session.name
    print(f"Using latest session: {session_id}")

    sample_plan = PlannerPlan(
        tasks=[
            PlannedTask(
                id="t_summary",
                kind="doc_summary",
                query="Summarise the key financial and legal terms of this loan agreement.",
                domain="finance",
                scope="global",
                language="en",
            ),
            PlannedTask(
                id="t_explain_repayment",
                kind="doc_explain",
                query="Explain the repayment schedule, interest, and late payment penalties "
                      "in simple borrower-friendly language.",
                domain="finance",
                scope="local",
                language="en",
                tone="borrower_friendly",
            ),
            PlannedTask(
                id="f_emi",
                kind="finance_emi",
                domain="finance",
                payload={
                    "principal": 10_000.0,
                    "annual_rate_percent": 8.0,
                    "tenure_months": 36,
                    "frequency": "monthly",
                    "currency": "USD",
                },
            ),
        ],
        final_answer_instructions=(
            "First, give a short summary of the loan agreement.\n"
            "Second, explain repayment and interest in simple borrower language.\n"
            "Third, clearly present the EMI calculation and totals.\n"
        ),
    )

    exec_results = execute_plan_on_session(
        plan=sample_plan,
        session_id=session_id,
        top_k_local=8,
    )

    final_text = build_final_answer(
        user_query="I uploaded this loan PDF. Please explain my loan and tell me the EMI for 10,000 at 8% for 36 months.",
        plan=sample_plan,
        doc_answers=exec_results["doc_answers"],
        finance_results=exec_results["finance_results"],
        output_language="en",
    )

    print("\n=========== FINAL ANSWER ===========\n")
    print(final_text)
    print("\n====================================\n")
