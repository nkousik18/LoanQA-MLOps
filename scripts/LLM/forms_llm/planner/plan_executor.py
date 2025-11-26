# scripts/LLM/forms_llm/planner/plan_executor.py

from __future__ import annotations

import os
import sys
import re
from typing import Dict, Any, List

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.LLM.forms_llm.planner.plan_schema import PlannerPlan, PlannedTask

# doc task machinery
from scripts.LLM.forms_llm.planner.doc_tasks import DocTask, DocTaskKind, DocScope
from scripts.LLM.forms_llm.planner.doc_task_executor import run_doc_tasks_on_document
from scripts.LLM.forms_llm.planner.doc_task_answer_llm import answer_many_doc_tasks

# finance machinery
from scripts.LLM.forms_llm.planner.finance_tasks import FinanceTask, FinanceTaskKind
from scripts.LLM.forms_llm.planner.finance_executor import run_finance_tasks


def _is_doc_task(t: PlannedTask) -> bool:
    return t.kind.startswith("doc_")


def _is_finance_task(t: PlannedTask) -> bool:
    return t.kind.startswith("finance_")


# ---------------------------------------------------------------------
# Finance helper: parse EMI numbers from full natural-language query
# ---------------------------------------------------------------------
# FIXED regex: capture full number (with optional commas) not just 1–3 digits
_NUM_RE = r"(\d[\d,]*)(?:\.\d+)?"


def _parse_emi_from_query(query: str) -> Dict[str, Any]:
    """
    Try to pull principal, annual_rate_percent, tenure_months from a sentence like:
      "Calculate EMI for 7000 at 9% for 24 months"
      "EMI for 10,000 @ 8 percent, tenure 3 years"
    Returns dict with any recovered fields.
    """
    q = (query or "").lower()
    out: Dict[str, Any] = {}

    # principal: look for "for 7000" or "principal 7000" or "amount 7000"
    m_principal = re.search(rf"(?:for|principal|amount|loan)\s+{_NUM_RE}", q)
    if m_principal:
        val = m_principal.group(1).replace(",", "")
        try:
            out["principal"] = float(val)
        except Exception:
            pass

    # rate: "at 9%" / "9 percent" / "interest 8%"
    m_rate = re.search(rf"{_NUM_RE}\s*(?:%|percent|percentage)", q)
    if m_rate:
        val = m_rate.group(1).replace(",", "")
        try:
            out["annual_rate_percent"] = float(val)
        except Exception:
            pass

    # tenure: "24 months" / "3 years"
    m_months = re.search(rf"{_NUM_RE}\s*(?:months|month|mos|mo)\b", q)
    if m_months:
        val = m_months.group(1).replace(",", "")
        try:
            out["tenure_months"] = int(float(val))
        except Exception:
            pass
    else:
        m_years = re.search(rf"{_NUM_RE}\s*(?:years|year|yrs|yr)\b", q)
        if m_years:
            val = m_years.group(1).replace(",", "")
            try:
                out["tenure_months"] = int(float(val) * 12)
            except Exception:
                pass

    # optional extras
    if "weekly" in q:
        out["frequency"] = "weekly"
    elif "yearly" in q or "annual" in q:
        out["frequency"] = "yearly"
    else:
        out["frequency"] = "monthly"

    # currency guess
    if "inr" in q or "₹" in query:
        out["currency"] = "INR"
    elif "usd" in q or "$" in query:
        out["currency"] = "USD"

    return out


def execute_plan_on_session(
    plan: PlannerPlan,
    session_id: str,
    top_k_local: int = 8,
) -> Dict[str, Any]:
    """
    Execute a PlannerPlan for a single document session.

    - doc_* tasks -> DocTask + numpy RAG + Groq answers
    - finance_* tasks -> FinanceTask + EMI tool (no LLM)

    Important:
      - If finance payload is missing, we try parsing numbers from finance query.
      - If still missing, we DO NOT crash. We return a 'needs_input' result.
    """
    # -----------------------------
    # 1) Split tasks by kind
    # -----------------------------
    doc_planned: List[PlannedTask] = [t for t in plan.tasks if _is_doc_task(t)]
    fin_planned: List[PlannedTask] = [t for t in plan.tasks if _is_finance_task(t)]

    # -----------------------------
    # 2) Build DocTask objects
    # -----------------------------
    doc_tasks: List[DocTask] = []
    for t in doc_planned:
        try:
            kind = DocTaskKind(t.kind)
        except ValueError:
            raise ValueError(f"Unknown doc task kind: {t.kind}")

        if t.scope is None:
            raise ValueError(f"Doc task {t.id} missing 'scope' (local/global)")

        try:
            scope = DocScope(t.scope)
        except ValueError:
            raise ValueError(f"Invalid scope for doc task {t.id}: {t.scope}")

        doc_tasks.append(
            DocTask(
                id=t.id,
                kind=kind,
                query=t.query or "",
                scope=scope,
                language=t.language or "en",
                tone=t.tone,
            )
        )

    # -----------------------------
    # 3) Build FinanceTask objects (robust)
    # -----------------------------
    runnable_finance_tasks: List[FinanceTask] = []
    pending_finance_results: Dict[str, Any] = {}

    for t in fin_planned:
        if t.kind != "finance_emi":
            pending_finance_results[t.id] = {
                "kind": t.kind,
                "status": "unsupported_kind",
                "note": f"Unsupported finance task kind: {t.kind}",
            }
            continue

        payload = (t.payload or {}).copy()

        # If payload empty, try parsing from finance query sentence
        if not payload:
            payload.update(_parse_emi_from_query(t.query or ""))

        principal = payload.get("principal")
        annual_rate = payload.get("annual_rate_percent")
        tenure_months = payload.get("tenure_months")

        missing = []
        if principal is None or float(principal) <= 0:
            missing.append("principal")
        if annual_rate is None or float(annual_rate) <= 0:
            missing.append("annual_rate_percent")
        if tenure_months is None or int(tenure_months) <= 0:
            missing.append("tenure_months")

        if missing:
            pending_finance_results[t.id] = {
                "kind": "finance_emi",
                "status": "needs_input",
                "missing_fields": missing,
                "parsed_payload": payload,
                "note": (
                    "EMI calculation needs more numbers. "
                    f"Missing: {', '.join(missing)}."
                ),
            }
            continue

        runnable_finance_tasks.append(
            FinanceTask(
                id=t.id,
                kind=FinanceTaskKind.EMI,
                principal=float(principal),
                annual_rate_percent=float(annual_rate),
                tenure_months=int(tenure_months),
                frequency=payload.get("frequency", "monthly"),
                currency=payload.get("currency", "USD"),
            )
        )

    # -----------------------------
    # 4) Run document side
    # -----------------------------
    doc_answers: Dict[str, Any] = {}
    if doc_tasks:
        ctx_dict = run_doc_tasks_on_document(
            tasks=doc_tasks,
            session_id=session_id,
            top_k_local=top_k_local,
        )
        doc_answers = answer_many_doc_tasks(doc_tasks, ctx_dict)

    # -----------------------------
    # 5) Run finance side
    # -----------------------------
    finance_results: Dict[str, Any] = {}
    if runnable_finance_tasks:
        raw_fin = run_finance_tasks(runnable_finance_tasks)
        finance_results = {
            task_id: {
                "kind": res.kind.value,
                "currency": res.currency,
                "payment_amount": res.payment_amount,
                "total_paid": res.total_paid,
                "total_interest": res.total_interest,
                "tenure_months": res.tenure_months,
                "annual_rate_percent": res.annual_rate_percent,
                "raw": res.raw,
                "status": "ok",
            }
            for task_id, res in raw_fin.items()
        }

    finance_results.update(pending_finance_results)

    return {
        "plan": plan.to_dict(),
        "doc_answers": doc_answers,
        "finance_results": finance_results,
    }


if __name__ == "__main__":
    from pathlib import Path

    sessions_dir = Path(PROJECT_ROOT) / "data" / "local_pipeline" / "sessions"
    session_dirs = [
        d for d in sessions_dir.iterdir()
        if d.is_dir() and d.name.startswith("session_")
    ]
    if not session_dirs:
        print(f"No session_* folders found in {sessions_dir}")
        sys.exit(1)

    latest_session = max(session_dirs, key=lambda d: d.stat().st_mtime)
    test_session_id = latest_session.name
    print(f"Using latest session: {test_session_id}")

    from scripts.LLM.forms_llm.planner.plan_schema import PlannerPlan, PlannedTask

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
                query="Calculate EMI for 10000 at 8% for 36 months.",
                domain="finance",
                payload={},  # intentionally empty to test parsing
            ),
        ],
        final_answer_instructions=(
            "First, provide a short English summary of the loan agreement.\n"
            "Second, explain the repayment and interest in borrower-friendly language.\n"
            "Third, present the EMI calculation (monthly payment, total paid, total interest)."
        ),
    )

    results = execute_plan_on_session(
        plan=sample_plan,
        session_id=test_session_id,
        top_k_local=8,
    )

    print("\n===== DOC TASK RESULTS =====\n")
    for tid, ans in results["doc_answers"].items():
        print(f"\n--- Task {tid} ({ans['kind']}) ---")
        print(ans["answer"])

    print("\n===== FINANCE TASK RESULTS =====\n")
    for tid, res in results["finance_results"].items():
        print(f"\n--- Task {tid} ({res['kind']}) status={res.get('status')} ---")
        print(res)
