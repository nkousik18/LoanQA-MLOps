"""
LLMquery/prompts/prompt_router.py
Hybrid semantic–keyword–pattern–memory router for LoanDocQA+
"""

import re
import torch
from sentence_transformers import SentenceTransformer, util
from langchain_core.documents import Document

from LLMquery.prompts.finance_prompt import finance_prompt
from LLMquery.prompts.summary_prompt import summary_prompt
from LLMquery.prompts.translation_prompt import translation_prompt
from LLMquery.prompts.retrieval_prompt import retrieval_prompt
from LLMquery.prompts.explanation_prompt import explanation_prompt

# ------------------------------------------------------------
# Load lightweight sentence-transformer once (cached)
router_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ------------------------------------------------------------
# Intent prototype definitions
INTENT_EXAMPLES = {
    "finance": [
        "calculate interest", "loan repayment", "emi amount", "simple interest",
        "compound interest", "financial formula", "repayment schedule",
        "borrowed money", "loan calculator", "roi computation", "npv", "irr",
        "amount payable after time", "principal plus interest", "annual rate"
    ],
    "summary": [
        "summarize", "give main points", "key points", "overview",
        "abstract", "brief", "short summary", "outline", "summarize document"
    ],
    "translation": [
        "translate", "in spanish", "in hindi", "in french", "meaning of",
        "how do you say", "word for", "translation", "convert language"
    ],
    "explanation": [
        "explain", "define", "describe", "difference between",
        "what is", "how does it work", "purpose of", "explain concept"
    ],
    "retrieval": [
        "when", "where", "who", "what documents", "requirements",
        "eligibility", "information from document", "details about",
        "find clause", "look up", "get details"
    ]
}

# ------------------------------------------------------------
# Precompute mean embedding per intent
INTENT_MAP = {
    k: torch.mean(router_model.encode(v, convert_to_tensor=True), dim=0)
    for k, v in INTENT_EXAMPLES.items()
}

# ------------------------------------------------------------
# Keyword lists for lightweight lexical scoring
KEYWORDS = {k: [p.split()[0] for p in v] for k, v in INTENT_EXAMPLES.items()}


def keyword_score(question: str, intent: str) -> float:
    """Compute keyword overlap score (0–1)."""
    q_lower = question.lower()
    hits = sum(kw in q_lower for kw in KEYWORDS[intent])
    return min(hits / 3, 1.0)


def numeric_pattern_score(question: str) -> float:
    """
    Returns a numeric pattern confidence (0–1),
    detecting numbers, %, currencies, or time-related expressions
    to bias toward 'finance'
    """
    has_number = bool(re.search(r"\d+", question))
    has_percent = bool(re.search(r"%|percent", question.lower()))
    has_currency = bool(re.search(r"\$|€|eur|rs|₹", question.lower()))
    has_time = bool(re.search(r"\byear|month|day|week\b", question.lower()))
    signal_count = sum([has_number, has_percent, has_currency, has_time])
    return signal_count / 4.0


def detect_intent(question: str, last_intent: str = None):
    """
    Multi-feature hybrid intent detector combining semantic,
    keyword and numeric cues.
    Returns: (intent, confidence, gap)
    """
    q_vec = router_model.encode(question, convert_to_tensor=True)
    sem_scores = {k: float(util.cos_sim(q_vec, v)) for k, v in INTENT_MAP.items()}
    best, second = sorted(sem_scores.items(), key=lambda x: x[1], reverse=True)[:2]

    sem = best[1]
    kw = keyword_score(question, best[0])
    num = numeric_pattern_score(question)

    # Weighted hybrid (semantic 0.6, keyword 0.2, numeric 0.2)
    hybrid = 0.6 * sem + 0.2 * kw + 0.2 * num

    # Reinforce continuity from conversation memory
    if last_intent and best[0] == last_intent:
        hybrid += 0.05

    # Bias correction: numeric features push weak queries toward finance
    if best[0] != "finance" and num >= 0.6 and sem < 0.55:
        best = ("finance", sem + 0.05)

    gap = best[1] - second[1]
    intent = best[0] if hybrid >= 0.45 else "retrieval"

    return intent, round(hybrid, 3), round(gap, 3)


def safe_extract_context(docs):
    """
    Build multi-document context safely.
    Handles both LangChain Document and plain string inputs.
    """
    if not docs:
        return ""

    context_parts = []
    for d in docs:
        if isinstance(d, Document):
            content = d.page_content.strip()
        elif isinstance(d, str):
            content = d.strip()
        elif hasattr(d, "page_content"):
            # Cover SimpleNamespace and custom wrappers
            content = str(d.page_content).strip()
        else:
            content = str(d)
        if content:
            context_parts.append(content)
    return "\n\n".join(context_parts)


def build_prompt(question: str, docs, mode: str = None, conversation_history=None):
    """
    Context-aware, adaptive prompt builder.
    Returns (prompt_text, detected_intent, confidence, gap)
    """
    context = safe_extract_context(docs)

    last_intent = (
        conversation_history[-2]["intent"]
        if conversation_history and len(conversation_history) > 1
        else None
    )

    # Manual override
    if mode:
        intent, conf, gap = mode.lower(), 1.0, 1.0
    else:
        intent, conf, gap = detect_intent(question, last_intent)

    # Intent → Prompt template map
    prompt_map = {
        "finance": finance_prompt,
        "summary": summary_prompt,
        "translation": translation_prompt,
        "explanation": explanation_prompt,
        "retrieval": retrieval_prompt,
    }

    prompt_func = prompt_map.get(intent, retrieval_prompt)
    prompt = prompt_func(question, context)

    print(f"[Router] Intent: {intent} | Confidence: {conf} | Gap: {gap}")
    print(f"[Router] Context length: {len(context)} characters")

    return prompt, intent, conf, gap
