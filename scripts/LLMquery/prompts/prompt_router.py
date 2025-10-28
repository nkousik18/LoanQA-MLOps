"""
scripts/LLMquery/prompts/prompt_router.py
Hybrid semantic–keyword–pattern–memory router for LoanDocQA+
"""

import re
import torch
import time
from sentence_transformers import SentenceTransformer, util
from langchain_core.documents import Document

from scripts.extraction_pipeline.config import setup_logger
from scripts.LLMquery.prompts.finance_prompt import finance_prompt
from scripts.LLMquery.prompts.summary_prompt import summary_prompt
from scripts.LLMquery.prompts.translation_prompt import translation_prompt
from scripts.LLMquery.prompts.retrieval_prompt import retrieval_prompt
from scripts.LLMquery.prompts.explanation_prompt import explanation_prompt

# ============================================================
# Initialize Centralized Logger
# ============================================================
logger = setup_logger(__name__, log_type="llm")

# ============================================================
# Load Sentence Transformer (cached model)
# ============================================================
router_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
logger.info("🧠 Prompt Router initialized with model: sentence-transformers/all-MiniLM-L6-v2")

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
logger.debug("Computing prototype intent embeddings for router...")
INTENT_MAP = {
    k: torch.mean(router_model.encode(v, convert_to_tensor=True), dim=0)
    for k, v in INTENT_EXAMPLES.items()
}
logger.info(f"✅ Loaded {len(INTENT_MAP)} intent categories: {list(INTENT_MAP.keys())}")

# ------------------------------------------------------------
# Keyword lists for lightweight lexical scoring
KEYWORDS = {k: [p.split()[0] for p in v] for k, v in INTENT_EXAMPLES.items()}


# ============================================================
# Helper Functions
# ============================================================
def keyword_score(question: str, intent: str) -> float:
    """Compute keyword overlap score (0–1)."""
    q_lower = question.lower()
    hits = sum(kw in q_lower for kw in KEYWORDS[intent])
    score = min(hits / 3, 1.0)
    logger.debug(f"[Keyword Score] {intent}: {score:.2f}")
    return score


def numeric_pattern_score(question: str) -> float:
    """Detects numeric, % or currency cues for financial biasing."""
    has_number = bool(re.search(r"\d+", question))
    has_percent = bool(re.search(r"%|percent", question.lower()))
    has_currency = bool(re.search(r"\$|€|eur|rs|₹", question.lower()))
    has_time = bool(re.search(r"\byear|month|day|week\b", question.lower()))
    signal_count = sum([has_number, has_percent, has_currency, has_time])
    score = signal_count / 4.0
    logger.debug(f"[Numeric Score] {score:.2f}")
    return score


def detect_intent(question: str, last_intent: str = None):
    """
    Multi-feature hybrid intent detector combining semantic, keyword,
    and numeric cues.
    Returns: (intent, confidence, gap)
    """
    start = time.time()
    q_vec = router_model.encode(question, convert_to_tensor=True)
    sem_scores = {k: float(util.cos_sim(q_vec, v)) for k, v in INTENT_MAP.items()}
    best, second = sorted(sem_scores.items(), key=lambda x: x[1], reverse=True)[:2]

    sem = best[1]
    kw = keyword_score(question, best[0])
    num = numeric_pattern_score(question)
    hybrid = 0.6 * sem + 0.2 * kw + 0.2 * num

    if last_intent and best[0] == last_intent:
        hybrid += 0.05

    if best[0] != "finance" and num >= 0.6 and sem < 0.55:
        best = ("finance", sem + 0.05)

    gap = best[1] - second[1]
    intent = best[0] if hybrid >= 0.45 else "retrieval"
    duration = round(time.time() - start, 3)

    logger.info(
        f"[Router] Q='{question[:60]}...' | Intent='{intent}' | Confidence={hybrid:.3f} "
        f"| Gap={gap:.3f} | Time={duration}s"
    )
    return intent, round(hybrid, 3), round(gap, 3)


def safe_extract_context(docs):
    """
    Build multi-document context safely.
    Handles both LangChain Document and plain string inputs.
    """
    if not docs:
        logger.debug("No documents provided for context.")
        return ""

    context_parts = []
    for d in docs:
        if isinstance(d, Document):
            content = d.page_content.strip()
        elif isinstance(d, str):
            content = d.strip()
        elif hasattr(d, "page_content"):
            content = str(d.page_content).strip()
        else:
            content = str(d)
        if content:
            context_parts.append(content)

    logger.debug(f"Extracted {len(context_parts)} context chunks.")
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

    if mode:
        intent, conf, gap = mode.lower(), 1.0, 1.0
        logger.debug(f"[Manual Mode] Intent manually overridden → {intent}")
    else:
        intent, conf, gap = detect_intent(question, last_intent)

    prompt_map = {
        "finance": finance_prompt,
        "summary": summary_prompt,
        "translation": translation_prompt,
        "explanation": explanation_prompt,
        "retrieval": retrieval_prompt,
    }

    prompt_func = prompt_map.get(intent, retrieval_prompt)
    prompt = prompt_func(question, context)

    logger.info(
        f"[Router Decision] Intent='{intent}' | Confidence={conf:.3f} | "
        f"Gap={gap:.3f} | ContextLen={len(context)} chars"
    )

    return prompt, intent, conf, gap
