"""
Hybrid Prompt Router for LoanDocAI (vLLM + RAG-Integrated)
-----------------------------------------------------------
✓ Hard-rule overrides preserved
✓ Semantic similarity + embeddings preserved
✓ RAG context now injected ALWAYS
✓ Intent templates still respected
"""

import time
import torch
from sentence_transformers import SentenceTransformer, util
from langchain_core.documents import Document

from app.prompts.finance import build_finance_prompt
from app.prompts.summary import build_summary_prompt
from app.prompts.translation import build_translation_prompt
from app.prompts.explanation import build_explanation_prompt
from app.prompts.retrieval import build_retrieval_prompt

from app.utils import log as logger


# ------------------------------------------------------------
# DEVICE
# ------------------------------------------------------------
def _get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _get_device()
logger.info(f"[PromptRouter] Using SentenceTransformer on: {DEVICE}")

router_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device=DEVICE,
)


# ------------------------------------------------------------
# INTENT EXEMPLARS
# ------------------------------------------------------------
INTENT_EXAMPLES = {
    "summary": ["summarize", "give a summary", "key points"],
    "explanation": ["explain", "meaning of", "define", "describe"],
    "retrieval": ["who is eligible", "lookup", "locate", "find"],
    "translation": ["translate", "into hindi", "into spanish"],
    "finance": ["interest", "emi", "loan repayment", "principal"]
}

INTENT_EMB = {
    intent: torch.mean(router_model.encode(samples, convert_to_tensor=True), dim=0)
    for intent, samples in INTENT_EXAMPLES.items()
}


# ------------------------------------------------------------
# HARD RULE OVERRIDES
# ------------------------------------------------------------
def hard_rules(q: str):
    q = q.lower().strip()

    if q.startswith("summarize") or "summary" in q:
        return "summary"
    if q.startswith("explain") or "meaning of" in q or "define" in q:
        return "explanation"
    if "translate" in q or "into hindi" in q or "into spanish" in q:
        return "translation"
    if any(term in q for term in ["emi", "interest", "%", "rate", "loan", "repayment"]):
        return "finance"
    if q.startswith(("who", "where", "when", "what")):
        return "retrieval"

    return None


# ------------------------------------------------------------
# SEMANTIC INTENT DETECTION
# ------------------------------------------------------------
def detect_intent(question: str):
    hr = hard_rules(question)
    if hr:
        logger.info(f"[PromptRouter-HardRule] '{question[:40]}...' → {hr}")
        return hr, 1.0, {"confidence": 1.0, "gap": 1.0}

    start = time.time()
    q_vec = router_model.encode(question, convert_to_tensor=True)

    scores = {intent: float(util.cos_sim(q_vec, emb)) for intent, emb in INTENT_EMB.items()}

    (best_intent, best_score), (second_intent, second_score) = sorted(
        scores.items(), key=lambda x: x[1], reverse=True
    )[:2]

    hybrid = 0.75 * best_score + 0.20 * second_score + 0.05
    intent = best_intent if hybrid >= 0.15 else "retrieval"

    meta = {
        "confidence": round(hybrid, 3),
        "gap": round(best_score - second_score, 3),
    }

    logger.info(
        f"[PromptRouter] Intent={intent} Conf={hybrid:.3f} "
        f"best={best_intent}:{best_score:.3f} gap={best_score-second_score:.3f} "
        f"time={time.time()-start:.3f}s"
    )

    return intent, hybrid, meta


# ------------------------------------------------------------
# SAFE CONTEXT EXTRACTOR
# ------------------------------------------------------------
def safe_extract_context(docs):
    if not docs:
        return ""
    if isinstance(docs, str):
        return docs

    parts = []
    for d in docs:
        if isinstance(d, Document):
            parts.append(d.page_content)
        else:
            parts.append(str(d))

    return "\n\n".join(parts)


# ------------------------------------------------------------
# UNIFIED RAG-AWARE PROMPT BUILDER
# ------------------------------------------------------------
def build_prompt(question, context_docs, retrieved_chunks=None, mode=None):
    context = safe_extract_context(context_docs)
    retrieved_chunks = retrieved_chunks or []

    # (1) Intent detection
    if mode:
        intent = mode.lower()
        conf = 1.0
        logger.info(f"[PromptRouter] Mode override → {intent}")
    else:
        intent, conf, _ = detect_intent(question)

    # (2) Intent → Template mapping
    PROMPTS = {
        "finance": build_finance_prompt,
        "summary": build_summary_prompt,
        "translation": build_translation_prompt,
        "explanation": build_explanation_prompt,
        "retrieval": build_retrieval_prompt,
    }

    fn = PROMPTS.get(intent, build_retrieval_prompt)

    # (3) ALWAYS pass context + chunks
    prompt = fn(
        question=question,
        context=context,
        retrieved_chunks=retrieved_chunks
    )

    logger.info(f"[PromptRouter] Prompt built | intent={intent} | ctx_len={len(context)}")

    return prompt, intent, conf
