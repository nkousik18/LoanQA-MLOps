"""
Improved Prompt Router for LoanDocQA+
- Hard-rule overrides
- Prefix-based intent detection
- Reduced numeric bias
- Balanced example embeddings
- GPU accelerated (CUDA/MPS/CPU fallback)
"""

import re
import os
import json
import time
import torch
import requests

from sentence_transformers import SentenceTransformer, util
from langchain_core.documents import Document

from scripts.extraction_pipeline.config import setup_logger
from scripts.LLMquery.prompts.finance_prompt import finance_prompt
from scripts.LLMquery.prompts.summary_prompt import summary_prompt
from scripts.LLMquery.prompts.translation_prompt import translation_prompt
from scripts.LLMquery.prompts.retrieval_prompt import retrieval_prompt
from scripts.LLMquery.prompts.explanation_prompt import explanation_prompt


# ============================================================
# Device Selection (Cross-platform)
# ============================================================

def _get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = _get_device()


# ============================================================
# Logger
# ============================================================
logger = setup_logger(__name__, log_type="llm")


# ============================================================
# Load embedding model
# ============================================================

router_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device=DEVICE
)

logger.info(f" Prompt Router initialized with MiniLM-L6-v2 on device: {DEVICE}")


OLLAMA_API_BASE_URL = os.getenv("OLLAMA_API_BASE_URL", "http://localhost:11434")


def check_ollama_connection():
    """Verify Ollama connectivity (once)."""
    try:
        r = requests.get(f"{OLLAMA_API_BASE_URL}/api/tags", timeout=5)
        if r.status_code == 200:
            print(f" Connected to Ollama at {OLLAMA_API_BASE_URL}")
            return True
    except Exception as e:
        print(f"⚠️ Ollama not reachable → {e}")
    return False


check_ollama_connection()


# ============================================================
# Intent Examples (Balanced)
# ============================================================

INTENT_EXAMPLES = {
    # trimmed & balanced
    "summary": [
        "summarize the document",
        "give an overview",
        "provide a short summary",
        "key points of the text",
        "brief outline",
    ],
    "explanation": [
        "explain the concept",
        "define this term",
        "difference between",
        "what does this mean",
        "describe this idea",
    ],
    "retrieval": [
        "when does",
        "who is eligible",
        "where is this mentioned",
        "what documents are required",
        "find this information",
    ],
    "translation": [
        "translate to spanish",
        "translate into hindi",
        "how to say this in",
        "word in french",
    ],
    "finance": [
        "interest rate",
        "calculate interest",
        "loan repayment",
        "emi calculation",
        "borrowed amount",
        "payment schedule",
    ],
}

# Precompute mean embeddings
INTENT_MAP = {
    k: torch.mean(router_model.encode(v, convert_to_tensor=True), dim=0)
    for k, v in INTENT_EXAMPLES.items()
}

logger.info(f" Loaded {len(INTENT_MAP)} intents.")


# ============================================================
# HARD RULES (Fixes the  misclassification issue)
# ============================================================

def hard_rules(question: str):
    q = question.lower().strip()

    if q.startswith("summarize") or "summary" in q:
        return "summary"

    if q.startswith("translate") or "into hindi" in q or "into spanish" in q:
        return "translation"

    if q.startswith("explain") or q.startswith("define") or "meaning of" in q:
        return "explanation"

    if q.startswith("who") or q.startswith("when") or q.startswith("where") or q.startswith("what") and "rate" not in q:
        return "retrieval"

    # strong finance triggers
    if any(x in q for x in ["calculate", "%", "interest", "emi", "rate"]):
        return "finance"

    return None


# ============================================================
# SEMANTIC + KEYWORD Hybrid Intent
# ============================================================

def detect_intent(question: str, last_intent=None):
    """Improved semantic router with soft weights + penalties."""

    # ----------- Hard Rule Override -----------------
    hr = hard_rules(question)
    if hr:
        logger.info(f"[Router-HardRule] Q='{question[:40]}...' → Intent={hr}")
        return hr, 1.0, {"confidence": 1.0, "gap": 1.0}

    start = time.time()
    q_vec = router_model.encode(question, convert_to_tensor=True)

    # semantic similarity
    sem_scores = {k: float(util.cos_sim(q_vec, v)) for k, v in INTENT_MAP.items()}
    best, second = sorted(sem_scores.items(), key=lambda x: x[1], reverse=True)[:2]

    # soft scoring (reduced numeric bias)
    hybrid = 0.75 * best[1] + 0.20 * second[1] + 0.05  # normalization bump

    # prefix penalty — prevents finance domination
    ql = question.lower()

    if ql.startswith("summarize") and best[0] != "summary":
        hybrid -= 0.25

    if ql.startswith("explain") and best[0] != "explanation":
        hybrid -= 0.25

    if "translate" in ql and best[0] != "translation":
        hybrid -= 0.25

    # fallback
    if hybrid < 0.15:
        intent = "retrieval"
    else:
        intent = best[0]

    duration = round(time.time() - start, 3)
    logger.info(
        f"[Router] Q='{question[:45]}...' | Intent='{intent}' | Conf={hybrid:.3f} | Time={duration}s"
    )

    # -------------------------------
    # FIX: Always return metadata dict
    # -------------------------------
    metadata = {
        "confidence": round(hybrid, 3),
        "gap": round(best[1] - second[1], 3)
    }

    return intent, round(hybrid, 3), metadata



# ============================================================
# Build Prompt
# ============================================================

PROMPT_MAP = {
    "finance": finance_prompt,
    "summary": summary_prompt,
    "translation": translation_prompt,
    "explanation": explanation_prompt,
    "retrieval": retrieval_prompt,
}


def safe_extract_context(docs):
    if not docs:
        return ""
    parts = []
    for d in docs:
        if isinstance(d, Document):
            parts.append(d.page_content)
        else:
            parts.append(str(d))
    return "\n\n".join(parts)


def build_prompt(question: str, docs, mode=None, conversation_history=None):
    """Unified prompt builder with corrected routing logic."""

    context = safe_extract_context(docs)
    last_intent = None
    if conversation_history and len(conversation_history) > 1:
        last_intent = conversation_history[-2]["intent"]

    if mode:
        intent = mode.lower()
        conf = 1.0
        gap = 1.0
    else:
        intent, conf, gap = detect_intent(question, last_intent)

    fn = PROMPT_MAP.get(intent, retrieval_prompt)
    prompt = fn(question, context)

    logger.info(
        f"[Router Decision] intent={intent} conf={conf:.3f} ctx_len={len(context)}"
    )

    return prompt, intent, conf, gap
