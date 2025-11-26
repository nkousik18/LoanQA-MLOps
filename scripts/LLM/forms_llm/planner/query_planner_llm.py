# scripts/LLM/forms_llm/planner/query_planner_llm.py

from __future__ import annotations

import os
import sys
import json
import re
from pathlib import Path
from typing import Any, Dict, List

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.LLM.forms_llm.planner.plan_schema import PlannerPlan
from scripts.LLM.forms_llm.llm_clients.groq_client import call_groq_chat


# =========================================================
# Prompt pack loader
# =========================================================
# ✅ prompts live here: scripts/LLM/prompts_form/
PROMPTS_DIR = Path(PROJECT_ROOT) / "scripts" / "LLM" / "prompts_form"
PLANNER_PROMPT_PATH = PROMPTS_DIR / "planner_prompt.txt"


def _load_planner_prompt(default_language: str = "en") -> str:
    """
    Load planner prompt from scripts/LLM/prompts_form/planner_prompt.txt.
    If missing, fall back to an embedded prompt that matches your rules.
    """
    if PLANNER_PROMPT_PATH.exists():
        return PLANNER_PROMPT_PATH.read_text(encoding="utf-8")

    return (
        "You are a planner for a loan-and-legal document assistant.\n"
        "Output STRICT JSON task plans only.\n"
        "Available kinds: doc_qa, doc_explain, doc_summary, doc_translate, finance_emi.\n"
        "Only supported finance kind: finance_emi.\n"
        "Do not add tasks user didn’t ask for.\n"
        "Definitions -> doc_qa unless user asks explain clearly -> doc_explain.\n"
        "Calculations: only EMI/mortgage monthly payment/installment -> finance_emi.\n"
        "Unsupported calc -> doc_qa fallback.\n"
        "doc_summary->global. doc_qa local by default, global for whole-doc QA.\n"
        "Top-level JSON keys: tasks, final_answer_instructions.\n"
    )


# =========================================================
# Keyword sets (match planner_prompt.txt)
# =========================================================
_NUM_PATTERN = re.compile(r"\d")

_SUMMARY_WORDS = ["summarise", "summarize", "summary", "overview", "brief"]
_EXPLAIN_WORDS = ["explain", "clarify", "break down", "help me understand", "details", "explain clearly"]
_TRANSLATE_WORDS = [
    "translate",
    "in telugu", "to telugu",
    "in hindi", "to hindi",
    "in tamil", "to tamil",
    "in spanish", "to spanish",
    "in french", "to french",
]

# definition phrases for ANY term (no numbers + no calc verbs)
_DEF_PHRASES = [
    "what is", "define", "meaning of", "what is meant by", "mean by",
    "stands for", "how does", "how do", "how is", "how it works",
]

# calc verbs (generic)
_CALC_VERBS = [
    "calculate", "calculation", "compute", "find", "estimate",
    "how much", "monthly payment", "installment", "instalment",
    "total payable", "total payment", "total amount to pay",
    "interest amount", "repayment amount",
]

# supported EMI-like calc signals
_EMI_CALC_SIGNALS = [
    "emi", "equated monthly installment", "equated monthly instalment",
    "monthly installment", "monthly instalment",
    "mortgage payment", "home loan payment", "loan payment",
    "installment", "instalment", "monthly payment",
]

# whole-doc QA signals for global scope
_WHOLEDOC_QA_SIGNALS = [
    "overall", "whole agreement", "whole document", "entire agreement", "entire document",
    "who are the parties", "parties involved",
    "effective date", "signing date",
    "list all", "all fees", "all penalties", "all obligations",
    "mention", "anywhere",
]


# =========================================================
# Small helpers
# =========================================================
def _wants_summary(user_query: str) -> bool:
    ql = (user_query or "").lower()
    return any(w in ql for w in _SUMMARY_WORDS)


def _wants_explain(user_query: str) -> bool:
    ql = (user_query or "").lower()
    return any(w in ql for w in _EXPLAIN_WORDS)


def _wants_translate(user_query: str) -> bool:
    ql = (user_query or "").lower()
    return any(w in ql for w in _TRANSLATE_WORDS)


def _infer_domain_from_query(ql: str) -> str | None:
    if any(w in ql for w in ["fee", "fees", "interest", "apr", "emi", "repayment",
                             "amount", "payable", "installment", "instalment",
                             "mortgage", "rate", "tenure", "principal"]):
        return "finance"
    if any(w in ql for w in ["default", "termination", "collateral", "guarantor",
                             "dispute", "governing law", "jurisdiction",
                             "obligation", "rights", "penalty"]):
        return "legal"
    if any(w in ql for w in ["translate", "language", "summary", "overview",
                             "parties", "effective date", "document about"]):
        return "general"
    return None


def _should_global_doc_qa(ql: str) -> bool:
    if any(sig in ql for sig in _WHOLEDOC_QA_SIGNALS):
        return True
    if "does this" in ql and ("agreement" in ql or "document" in ql) and "mention" in ql:
        return True
    return False


def _next_finance_id(existing_ids: set[str]) -> str:
    base = "f_emi"
    if base not in existing_ids:
        return base
    i = 2
    while f"{base}_{i}" in existing_ids:
        i += 1
    return f"{base}_{i}"


# =========================================================
# Definition vs calculation detectors (universal)
# =========================================================
def _has_definition_intent(user_query: str) -> bool:
    """
    Definition/meaning/process intent:
      - definition phrases present
      - NO numbers
      - NO calc verbs
    """
    ql = (user_query or "").lower().strip()
    if not ql:
        return False

    has_def_phrase = any(p in ql for p in _DEF_PHRASES)
    has_numbers = bool(_NUM_PATTERN.search(ql))
    has_calc_verbs = any(v in ql for v in _CALC_VERBS)

    return has_def_phrase and (not has_numbers) and (not has_calc_verbs)


def _has_calc_intent(user_query: str) -> bool:
    """
    Generic calc intent:
      - calc verbs present OR
      - numbers + finance/payment context

    BUT conceptual "how is X calculated?" (no numbers) is NOT a numeric calc intent.
    """
    ql = (user_query or "").lower().strip()
    if not ql:
        return False

    has_numbers = bool(_NUM_PATTERN.search(ql))
    has_calc_verbs = any(v in ql for v in _CALC_VERBS)

    # conceptual, no numbers -> NOT calc intent
    if (ql.startswith("how is") or ql.startswith("how does")) and has_calc_verbs and not has_numbers:
        return False

    if has_calc_verbs:
        return True

    if has_numbers and any(s in ql for s in ["emi", "interest", "payment", "installment",
                                            "instalment", "mortgage", "tenure", "rate"]):
        return True

    return False


def _is_supported_emi_calc(user_query: str) -> bool:
    """
    Only EMI/mortgage monthly payment/installment calculations are supported by tools.
    """
    ql = (user_query or "").lower()
    if not ql:
        return False
    if not _has_calc_intent(user_query):
        return False
    return any(sig in ql for sig in _EMI_CALC_SIGNALS)


# =========================================================
# Default fallback plan (only when planner LLM fails)
# =========================================================
def _default_plan(user_query: str, language: str = "en") -> PlannerPlan:
    """
    Matches planner_prompt.txt exactly.
    """
    from scripts.LLM.forms_llm.planner.plan_schema import PlannedTask

    ql = (user_query or "").lower().strip()

    tasks: List[PlannedTask] = []

    # ---- A) Definition vs explain-definition
    if _has_definition_intent(user_query):
        kind = "doc_explain" if _wants_explain(user_query) else "doc_qa"
        tasks.append(
            PlannedTask(
                id="t1",
                kind=kind,
                query=user_query.strip(),
                domain=_infer_domain_from_query(ql) or "general",
                scope="local",
                language=language,
                tone="borrower_friendly",
                depends_on=[],
            )
        )
        return PlannerPlan(
            tasks=tasks,
            final_answer_instructions=(
                "Explain clearly and relate to the agreement if stated."
                if kind == "doc_explain"
                else "Answer the definition directly."
            ),
        )

    # ---- B) Summary intent
    if _wants_summary(user_query):
        tasks.append(
            PlannedTask(
                id="t_summary",
                kind="doc_summary",
                query="Summarise the key financial and legal terms of this agreement.",
                domain="general",
                scope="global",
                language=language,
                tone=None,
                depends_on=[],
            )
        )

    # ---- C) Explain intent
    if _wants_explain(user_query):
        scope = "global" if any(x in ql for x in ["agreement", "document", "contract", "loan doc", "loan document"]) else "local"
        tasks.append(
            PlannedTask(
                id="t_explain",
                kind="doc_explain",
                query=(
                    "Explain the loan agreement or the specific section the user asked about "
                    "in plain borrower-friendly language. Do not summarise unless asked."
                ),
                domain=_infer_domain_from_query(ql) or "general",
                scope=scope,
                language=language,
                tone="borrower_friendly",
                depends_on=[],
            )
        )

    # ---- D) Translate intent
    if _wants_translate(user_query):
        scope = "global" if any(x in ql for x in ["whole", "entire", "full document", "agreement", "document"]) else "local"
        tasks.append(
            PlannedTask(
                id="t_translate",
                kind="doc_translate",
                query=user_query.strip() or "Translate the requested content.",
                domain="general",
                scope=scope,
                language=language,
                tone=None,
                depends_on=[],
            )
        )

    # ---- E) Calculations
    if _is_supported_emi_calc(user_query):
        tasks.append(
            PlannedTask(
                id="f1",
                kind="finance_emi",
                query=user_query,
                domain="finance",
                scope=None,
                language=language,
                tone=None,
                payload={},
                depends_on=[],
            )
        )
    elif _has_calc_intent(user_query):
        # unsupported calc fallback -> ONE doc_qa
        tasks.append(
            PlannedTask(
                id="t_calc_fallback",
                kind="doc_qa",
                query=(
                    "From the agreement, extract any stated result or the inputs needed to compute it "
                    "(principal, interest rate, tenure, fees). If missing, say so."
                ),
                domain="finance",
                scope="local",
                language=language,
                tone="borrower_friendly",
                depends_on=[],
            )
        )

    # ---- F) If nothing matched -> default QA
    if not tasks:
        tasks = [
            PlannedTask(
                id="t1",
                kind="doc_qa",
                query=user_query.strip() or "Answer the user's question from the document.",
                domain=_infer_domain_from_query(ql) or "general",
                scope="local",
                language=language,
                tone="borrower_friendly",
                depends_on=[],
            )
        ]

    return PlannerPlan(
        tasks=tasks,
        final_answer_instructions="Follow the tasks in order. Keep the answer focused on the user's query."
    )


# =========================================================
# Sanitization (enforce rules even if LLM drifts)
# =========================================================
_ALLOWED_KINDS = {"doc_qa", "doc_explain", "doc_summary", "doc_translate", "finance_emi"}
_KIND_MAP = {
    "doc_summarize": "doc_summary",
    "summary": "doc_summary",
    "summarise": "doc_summary",
    "doc_explanation": "doc_explain",
    "explain": "doc_explain",
    "qa": "doc_qa",
    "doc_question_answer": "doc_qa",
    "emi": "finance_emi",
    "mortgage": "finance_emi",
}

_ALLOWED_DOMAINS = {"finance", "legal", "general"}
_ALLOWED_SCOPES = {"local", "global"}
_ALLOWED_TONES = {"borrower_friendly", "expert"}


def _sanitize_plan(plan: PlannerPlan, user_query: str, default_language: str) -> PlannerPlan:
    """
    Enforce your strict planning rules on the LLM output.
    """
    from scripts.LLM.forms_llm.planner.plan_schema import PlannedTask

    ql = (user_query or "").lower().strip()

    wants_summary = _wants_summary(user_query)
    wants_explain = _wants_explain(user_query)
    wants_translate = _wants_translate(user_query)

    def_intent = _has_definition_intent(user_query)
    calc_intent = _has_calc_intent(user_query)
    supported_emi_calc = _is_supported_emi_calc(user_query)

    cleaned: List[PlannedTask] = []

    # ---- 1) Clean + drop drifted tasks
    for t in plan.tasks:
        kind_raw = (t.kind or "").strip()
        kind = _KIND_MAP.get(kind_raw, kind_raw)
        if kind not in _ALLOWED_KINDS:
            continue

        # Drop kinds not asked
        if kind == "doc_summary" and not wants_summary:
            continue
        if kind == "doc_explain" and not wants_explain:
            continue
        if kind == "doc_translate" and not wants_translate:
            continue

        domain = t.domain if (t.domain in _ALLOWED_DOMAINS) else None
        scope = t.scope if (t.scope in _ALLOWED_SCOPES) else None
        tone = t.tone if (t.tone in _ALLOWED_TONES) else None
        language = t.language or default_language

        # Defaults per kind (exactly per prompt)
        if kind == "finance_emi":
            domain = "finance"
            scope = None
            tone = None

        elif kind == "doc_summary":
            domain = domain or "general"
            scope = "global"
            tone = None

        elif kind == "doc_translate":
            domain = domain or "general"
            if scope not in {"local", "global"}:
                scope = "global" if any(x in ql for x in ["whole", "entire", "full document", "agreement", "document"]) else "local"
            tone = None

        elif kind == "doc_explain":
            domain = domain or (_infer_domain_from_query(ql) or "general")
            if scope not in {"local", "global"}:
                scope = "global" if any(x in ql for x in ["agreement", "document", "contract", "loan doc", "loan document"]) else "local"
            tone = tone or "borrower_friendly"

        else:  # doc_qa
            domain = domain or (_infer_domain_from_query(ql) or "general")
            if scope not in {"local", "global"}:
                scope = "global" if _should_global_doc_qa(ql) else "local"
            tone = tone or "borrower_friendly"

        payload = t.payload if isinstance(t.payload, dict) else {}
        depends_on = t.depends_on if isinstance(t.depends_on, list) else []

        cleaned.append(
            PlannedTask(
                id=t.id or f"task_{len(cleaned)+1}",
                kind=kind,
                query=(t.query or user_query).strip(),
                domain=domain,
                scope=scope,
                language=language,
                tone=tone,
                depends_on=depends_on,
                payload=payload,
            )
        )

    # ---- 2) Universal definition rule
    if def_intent:
        if wants_explain:
            cleaned = [t for t in cleaned if t.kind == "doc_explain"]
            if not cleaned:
                cleaned = [
                    PlannedTask(
                        id="t1",
                        kind="doc_explain",
                        query=user_query.strip(),
                        domain=_infer_domain_from_query(ql) or "general",
                        scope="local",
                        language=default_language,
                        tone="borrower_friendly",
                        depends_on=[],
                    )
                ]
            plan.tasks = cleaned
            plan.final_answer_instructions = "Explain clearly and relate to the agreement if stated."
            return plan
        else:
            cleaned = [t for t in cleaned if t.kind == "doc_qa"]
            if not cleaned:
                cleaned = [
                    PlannedTask(
                        id="t1",
                        kind="doc_qa",
                        query=user_query.strip(),
                        domain=_infer_domain_from_query(ql) or "general",
                        scope="local",
                        language=default_language,
                        tone="borrower_friendly",
                        depends_on=[],
                    )
                ]
            plan.tasks = cleaned
            plan.final_answer_instructions = "Answer the definition directly."
            return plan

    # ---- 3) Supported EMI calc -> ensure finance_emi exists
    if supported_emi_calc:
        if not any(t.kind == "finance_emi" for t in cleaned):
            fin_id = _next_finance_id({t.id for t in cleaned})
            cleaned.append(
                PlannedTask(
                    id=fin_id,
                    kind="finance_emi",
                    query=user_query,
                    domain="finance",
                    scope=None,
                    language=default_language,
                    tone=None,
                    payload={},
                    depends_on=[],
                )
            )

    # ---- 4) Unsupported calc -> NEVER allow finance tasks
    if calc_intent and not supported_emi_calc:
        cleaned = [t for t in cleaned if t.kind != "finance_emi"]

        # ensure ONE doc_qa fallback exists
        if not any(t.kind == "doc_qa" for t in cleaned):
            cleaned.append(
                PlannedTask(
                    id="t_calc_fallback",
                    kind="doc_qa",
                    query=(
                        "From the agreement, extract any stated result or the inputs needed to compute it "
                        "(principal, interest rate, tenure, fees). If missing, say so."
                    ),
                    domain="finance",
                    scope="local",
                    language=default_language,
                    tone="borrower_friendly",
                    depends_on=[],
                )
            )

    # ---- 5) If empty -> hard fallback
    if not cleaned:
        return _default_plan(user_query, language=default_language)

    # ---- 6) Stable ordering
    priority = {"doc_summary": 0, "doc_explain": 1, "doc_qa": 2, "doc_translate": 3, "finance_emi": 4}
    cleaned.sort(key=lambda t: priority.get(t.kind, 99))

    plan.tasks = cleaned
    if not (plan.final_answer_instructions or "").strip():
        plan.final_answer_instructions = "Follow the tasks in order. Keep the answer focused on the user's query."
    return plan


# =========================================================
# Planner LLM
# =========================================================
def plan_user_query_with_llm(
    user_query: str,
    default_language: str = "en",
) -> PlannerPlan:

    system_prompt = _load_planner_prompt(default_language=default_language)

    user_prompt = (
        f"User query:\n{user_query}\n\n"
        "Decide what the assistant should do.\n"
        f"If the user does not specify a language, default language is '{default_language}'.\n\n"
        "Now output the TASK PLAN as JSON.\n"
    )

    raw = call_groq_chat(system_prompt, user_prompt)

    try:
        data: Dict[str, Any] = json.loads(raw)
        plan = PlannerPlan.from_dict(data)
    except Exception:
        return _default_plan(user_query, language=default_language)

    plan = _sanitize_plan(plan, user_query, default_language)
    return plan


# =========================================================
# Manual test
# =========================================================
if __name__ == "__main__":
    examples = [
        "Explain my loan agreement.",
        "What is APR?",
        "Explain clearly APR.",
        "Calculate EMI for 10000 at 8% for 36 months.",
        "Calculate total payable for this loan.",
        "List all fees and penalties in this agreement.",
        "Explain the default clause.",
        "Give me a summary of this loan.",
        "Explain this agreement in Telugu and also calculate EMI for 7000 at 9% for 24 months.",
    ]

    for q in examples:
        print("\n==============================")
        print("USER QUERY:", q)
        plan = plan_user_query_with_llm(q, default_language="en")
        print("\nGenerated plan JSON:\n")
        print(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2))
