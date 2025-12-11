# scripts/LLM/forms_llm/planner/query_planner_llm.py

from __future__ import annotations

import os
import sys
import json
import re
from typing import Any, Dict, List

# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.LLM.forms_llm.planner.plan_schema import PlannerPlan, PlannedTask
from scripts.aws_extraction_scripts.log_utils import get_logger

LOGGER = get_logger(__name__)

# =========================================================
# Keyword sets (match the original planner_prompt rules)
# =========================================================
_NUM_PATTERN = re.compile(r"\d")

_SUMMARY_WORDS = ["summarise", "summarize", "summary", "overview", "brief"]
_EXPLAIN_WORDS = [
    "explain",
    "clarify",
    "break down",
    "help me understand",
    "details",
    "explain clearly",
]
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
    "what is",
    "define",
    "meaning of",
    "what is meant by",
    "mean by",
    "stands for",
    "how does",
    "how do",
    "how is",
    "how it works",
]

# calc verbs (generic)
_CALC_VERBS = [
    "calculate",
    "calculation",
    "compute",
    "find",
    "estimate",
    "how much",
    "monthly payment",
    "installment",
    "instalment",
    "total payable",
    "total payment",
    "total amount to pay",
    "interest amount",
    "repayment amount",
]

# supported EMI-like calc signals
_EMI_CALC_SIGNALS = [
    "emi",
    "equated monthly installment",
    "equated monthly instalment",
    "monthly installment",
    "monthly instalment",
    "mortgage payment",
    "home loan payment",
    "loan payment",
    "installment",
    "instalment",
    "monthly payment",
]

# whole-doc QA signals for global scope
_WHOLEDOC_QA_SIGNALS = [
    "overall",
    "whole agreement",
    "whole document",
    "entire agreement",
    "entire document",
    "who are the parties",
    "parties involved",
    "effective date",
    "signing date",
    "list all",
    "all fees",
    "all penalties",
    "all obligations",
    "mention",
    "anywhere",
]

# phrases that mean "look up value in THIS document", not compute a new one
_DOC_LOOKUP_PHRASES = [
    "in this document",
    "in our document",
    "in the document",
    "from this document",
    "from the document",
    "in this agreement",
    "in our agreement",
    "in the agreement",
    "from this agreement",
    "from the agreement",
    "in this loan document",
    "from this loan document",
    "in this loan agreement",
    "from this loan agreement",
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


def _mentions_doc_lookup(ql: str) -> bool:
    ql = ql.lower()
    return any(p in ql for p in _DOC_LOOKUP_PHRASES)


def _infer_domain_from_query(ql: str) -> str | None:
    if any(
        w in ql
        for w in [
            "fee",
            "fees",
            "interest",
            "apr",
            "emi",
            "repayment",
            "amount",
            "payable",
            "installment",
            "instalment",
            "mortgage",
            "rate",
            "tenure",
            "principal",
        ]
    ):
        return "finance"
    if any(
        w in ql
        for w in [
            "default",
            "termination",
            "collateral",
            "guarantor",
            "dispute",
            "governing law",
            "jurisdiction",
            "obligation",
            "rights",
            "penalty",
        ]
    ):
        return "legal"
    if any(
        w in ql
        for w in [
            "translate",
            "language",
            "summary",
            "overview",
            "parties",
            "effective date",
            "document about",
        ]
    ):
        return "general"
    return None


def _should_global_doc_qa(ql: str) -> bool:
    ql = ql.lower()
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
    if (
        (ql.startswith("how is") or ql.startswith("how does"))
        and has_calc_verbs
        and not has_numbers
    ):
        return False

    if has_calc_verbs:
        return True

    if has_numbers and any(
        s in ql
        for s in [
            "emi",
            "interest",
            "payment",
            "installment",
            "instalment",
            "mortgage",
            "tenure",
            "rate",
        ]
    ):
        return True

    return False


def _is_supported_emi_calc(user_query: str) -> bool:
    """
    Only EMI/mortgage monthly payment/installment calculations are supported by tools.

    We want to trigger this ONLY when the user clearly asks to CALCULATE EMI
    for a scenario (principal + rate + tenure), NOT when they just want to
    read EMI/interest that is already written in the document.
    """
    ql = (user_query or "").lower().strip()
    if not ql:
        return False

    # Must mention EMI / payment concepts at all
    if not any(sig in ql for sig in _EMI_CALC_SIGNALS):
        return False

    # If they explicitly say "in this document/agreement" -> treat as lookup, not calc
    if _mentions_doc_lookup(ql):
        return False

    # Need generic calc intent
    if not _has_calc_intent(user_query):
        return False

    # How many numeric tokens are present?
    num_count = len(_NUM_PATTERN.findall(ql))

    # Strong phrases that usually mean "calculate EMI", not "read from doc"
    strong_calc_phrases = [
        "calculate emi",
        "calculation of emi",
        "compute emi",
        "work out emi",
        "what will be my emi",
        "what would be my emi",
        "how much will my emi",
        "emi for ",
        "emi on ",
    ]

    if any(p in ql for p in strong_calc_phrases):
        return True

    # Fallback: if there are at least 2 numbers, it's likely a principal-rate-tenure scenario
    if num_count >= 2:
        return True

    # Otherwise, treat as doc lookup, not EMI tool
    return False


# =========================================================
# Default rule-based planner (no LLM)
# =========================================================
def _default_plan(user_query: str, language: str = "en") -> PlannerPlan:
    """
    Purely rule-based planner.

    Matches the logic of your old planner_prompt:
      - classify into doc_qa / doc_explain / doc_summary / doc_translate / finance_emi
      - choose local vs global scope
      - pick finance/legal/general domain
    """
    ql = (user_query or "").lower().strip()

    wants_summary = _wants_summary(user_query)
    wants_explain = _wants_explain(user_query)
    wants_translate = _wants_translate(user_query)
    def_intent = _has_definition_intent(user_query)
    calc_intent = _has_calc_intent(user_query)
    supported_emi = _is_supported_emi_calc(user_query)

    LOGGER.info(
        "[query_planner] _default_plan called | "
        f"language={language}, def_intent={def_intent}, "
        f"wants_summary={wants_summary}, wants_explain={wants_explain}, "
        f"wants_translate={wants_translate}, calc_intent={calc_intent}, "
        f"supported_emi={supported_emi}"
    )

    tasks: List[PlannedTask] = []

    # ---- A) Definition vs explain-definition
    if def_intent:
        kind = "doc_explain" if wants_explain else "doc_qa"
        LOGGER.info(
            f"[query_planner] Definition intent detected → kind={kind}, scope=local"
        )
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
        plan = PlannerPlan(
            tasks=tasks,
            final_answer_instructions=(
                "Explain clearly and relate to the agreement if stated."
                if kind == "doc_explain"
                else "Answer the definition directly."
            ),
        )
        LOGGER.info(
            "[query_planner] _default_plan produced 1 definition task: "
            f"{[t.kind for t in tasks]}"
        )
        return plan

    # ---- B) Summary intent
    if wants_summary:
        LOGGER.info("[query_planner] Summary intent detected → adding doc_summary")
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
    if wants_explain:
        scope = (
            "global"
            if any(
                x in ql
                for x in [
                    "agreement",
                    "document",
                    "contract",
                    "loan doc",
                    "loan document",
                ]
            )
            else "local"
        )
        LOGGER.info(
            f"[query_planner] Explain intent detected → adding doc_explain, scope={scope}"
        )
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
    if wants_translate:
        scope = (
            "global"
            if any(
                x in ql
                for x in ["whole", "entire", "full document", "agreement", "document"]
            )
            else "local"
        )
        LOGGER.info(
            f"[query_planner] Translate intent detected → adding doc_translate, scope={scope}"
        )
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
    if supported_emi:
        LOGGER.info(
            "[query_planner] Supported EMI calc detected → adding finance_emi task"
        )
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
    elif calc_intent:
        # unsupported calc fallback -> ONE doc_qa
        LOGGER.info(
            "[query_planner] Calc intent but not supported EMI → adding doc_qa fallback"
        )
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
        LOGGER.info(
            "[query_planner] No explicit intent matched → defaulting to single doc_qa"
        )
        tasks = [
            PlannedTask(
                id="t1",
                kind="doc_qa",
                query=user_query.strip()
                or "Answer the user's question from the document.",
                domain=_infer_domain_from_query(ql) or "general",
                scope="local",
                language=language,
                tone="borrower_friendly",
                depends_on=[],
            )
        ]

    plan = PlannerPlan(
        tasks=tasks,
        final_answer_instructions=(
            "Follow the tasks in order. Keep the answer focused on the user's query."
        ),
    )
    LOGGER.info(
        "[query_planner] _default_plan produced tasks: "
        f"{[t.kind for t in tasks]}"
    )
    return plan


# =========================================================
# Sanitization (still useful if you ever swap planner source)
# =========================================================
_ALLOWED_KINDS = {
    "doc_qa",
    "doc_explain",
    "doc_summary",
    "doc_translate",
    "finance_emi",
}
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


def _sanitize_plan(
    plan: PlannerPlan, user_query: str, default_language: str
) -> PlannerPlan:
    """
    Enforce strict planning rules on the plan output.
    Works even if the plan came from rules (now) or LLM (future).
    """
    ql = (user_query or "").lower().strip()

    wants_summary = _wants_summary(user_query)
    wants_explain = _wants_explain(user_query)
    wants_translate = _wants_translate(user_query)

    def_intent = _has_definition_intent(user_query)
    calc_intent = _has_calc_intent(user_query)
    supported_emi_calc = _is_supported_emi_calc(user_query)

    LOGGER.info(
        "[query_planner] _sanitize_plan start | "
        f"tasks_in={len(plan.tasks)}, wants_summary={wants_summary}, "
        f"wants_explain={wants_explain}, wants_translate={wants_translate}, "
        f"def_intent={def_intent}, calc_intent={calc_intent}, "
        f"supported_emi_calc={supported_emi_calc}"
    )

    cleaned: List[PlannedTask] = []

    # ---- 1) Clean + drop drifted tasks
    for t in plan.tasks:
        kind_raw = (t.kind or "").strip()
        kind = _KIND_MAP.get(kind_raw, kind_raw)
        if kind not in _ALLOWED_KINDS:
            LOGGER.info(
                f"[query_planner] Dropping task id={t.id} kind={kind_raw} "
                f"(mapped={kind}) – not allowed"
            )
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

        # Defaults per kind
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
                scope = (
                    "global"
                    if any(
                        x in ql
                        for x in [
                            "whole",
                            "entire",
                            "full document",
                            "agreement",
                            "document",
                        ]
                    )
                    else "local"
                )
            tone = None

        elif kind == "doc_explain":
            domain = domain or (_infer_domain_from_query(ql) or "general")
            if scope not in {"local", "global"}:
                scope = (
                    "global"
                    if any(
                        x in ql
                        for x in [
                            "agreement",
                            "document",
                            "contract",
                            "loan doc",
                            "loan document",
                        ]
                    )
                    else "local"
                )
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
                id=t.id or f"task_{len(cleaned) + 1}",
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
            plan.final_answer_instructions = (
                "Explain clearly and relate to the agreement if stated."
            )
            LOGGER.info(
                "[query_planner] _sanitize_plan finished (definition-explain path); "
                f"tasks={len(cleaned)}, kinds={[t.kind for t in cleaned]}"
            )
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
            LOGGER.info(
                "[query_planner] _sanitize_plan finished (definition-QA path); "
                f"tasks={len(cleaned)}, kinds={[t.kind for t in cleaned]}"
            )
            return plan

    # ---- 3) Supported EMI calc -> ensure finance_emi exists
    if supported_emi_calc:
        if not any(t.kind == "finance_emi" for t in cleaned):
            fin_id = _next_finance_id({t.id for t in cleaned})
            LOGGER.info(
                f"[query_planner] Sanitizer adding missing finance_emi task id={fin_id}"
            )
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
            LOGGER.info(
                "[query_planner] Sanitizer adding doc_qa calc fallback for unsupported calc"
            )
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
        LOGGER.info(
            "[query_planner] Sanitizer produced empty plan → falling back to _default_plan"
        )
        return _default_plan(user_query, language=default_language)

    # ---- 6) Stable ordering
    priority = {
        "doc_summary": 0,
        "doc_explain": 1,
        "doc_qa": 2,
        "doc_translate": 3,
        "finance_emi": 4,
    }
    cleaned.sort(key=lambda t: priority.get(t.kind, 99))

    plan.tasks = cleaned
    if not (plan.final_answer_instructions or "").strip():
        plan.final_answer_instructions = (
            "Follow the tasks in order. Keep the answer focused on the user's query."
        )

    LOGGER.info(
        "[query_planner] _sanitize_plan finished | "
        f"tasks_out={len(cleaned)}, kinds={[t.kind for t in cleaned]}"
    )
    return plan


# =========================================================
# Planner (NOW PURELY RULE-BASED, NO LLM CALL)
# =========================================================
def plan_user_query_with_llm(
    user_query: str,
    default_language: str = "en",
) -> PlannerPlan:
    """
    Deterministic planner:
      - NO external LLM call
      - Uses _default_plan (rules) + _sanitize_plan
      - Signature unchanged so loan_assistant_demo & Streamlit keep working
    """
    LOGGER.info(
        "[query_planner] plan_user_query_with_llm called | "
        f"default_language={default_language}, query_len={len(user_query or '')}"
    )
    base_plan = _default_plan(user_query, language=default_language)
    cleaned_plan = _sanitize_plan(base_plan, user_query, default_language)
    LOGGER.info(
        "[query_planner] plan_user_query_with_llm complete | "
        f"tasks={len(cleaned_plan.tasks)}, kinds={[t.kind for t in cleaned_plan.tasks]}"
    )
    return cleaned_plan


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
        "Does this agreement mention any prepayment penalty anywhere?",
        "Explain what is interest rate and EMI in this document.",
    ]

    for q in examples:
        print("\n==============================")
        print("USER QUERY:", q)
        plan = plan_user_query_with_llm(q, default_language="en")
        print("\nGenerated plan JSON:\n")
        print(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2))
