"""
scripts/model_selection/metrics.py
Improved scoring for RAG evaluation:
 - Groundedness (semantic + lexical)
 - Hallucination severity
 - Confidence
 - Summary divergence
"""

import re
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer, util
import torch


# ============================================================
# GLOBAL EMBEDDER (GPU/CPU adaptive)
# ============================================================

def _get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _get_device()

EMBEDDER = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device=DEVICE
)


# ============================================================
# Helper: join retrieved chunk text
# ============================================================

def _combine_retrieved_text(chunks):
    if not chunks:
        return ""
    return " ".join(c["text"] for c in chunks)


# ============================================================
# 1. Groundedness Score (0–1)
#    Measures how much of LLM output semantically aligns
#    with retrieved context.
# ============================================================

def groundedness_score(output: str, retrieved_chunks) -> float:
    if not output.strip() or not retrieved_chunks:
        return 0.0

    ctx = _combine_retrieved_text(retrieved_chunks)

    out_emb = EMBEDDER.encode(output, convert_to_tensor=True, device=DEVICE)
    ctx_emb = EMBEDDER.encode(ctx, convert_to_tensor=True, device=DEVICE)

    score = float(util.cos_sim(out_emb, ctx_emb))

    # normalize to 0–1 range
    return max(0.0, min((score + 1) / 2, 1.0))


# ============================================================
# 2. Hallucination Severity (0–1)
#    Measures factual drift: #out-of-context tokens
# ============================================================

def hallucination_severity(output: str, retrieved_chunks) -> float:
    if not output.strip():
        return 1.0

    ctx = _combine_retrieved_text(retrieved_chunks).lower()
    ctx_tokens = set(ctx.split())

    out_tokens = output.lower().split()

    if not ctx_tokens:
        return 1.0

    missing = [t for t in out_tokens if t not in ctx_tokens]
    sev = len(missing) / max(len(out_tokens), 1)

    return min(max(sev, 0.0), 1.0)


# ============================================================
# 3. Confidence Score (0–1)
#    Simple heuristic based on sentence structure.
# ============================================================

def confidence_score(output: str) -> float:
    if not output.strip():
        return 0.0

    long_sentences = sum(1 for s in output.split(".") if len(s.split()) > 18)
    score = 1 - long_sentences * 0.07

    return max(0.25, min(score, 1.0))


# ============================================================
# 4. Summary Divergence (0–1)
#    Measures similarity between summary and full doc context.
# ============================================================

def summary_divergence(output: str, retrieved_chunks) -> float:
    ctx = _combine_retrieved_text(retrieved_chunks)
    if not ctx.strip() or not output.strip():
        return 1.0
    sim = SequenceMatcher(None, output.lower(), ctx.lower()).ratio()
    return 1 - sim


# ============================================================
# 5. Unified Score Output
# ============================================================

def detect_hallucination(output: str, retrieved_chunks):
    """
    Unified final score used in evaluate_by_intent pipeline.
    """

    g = groundedness_score(output, retrieved_chunks)
    sev = hallucination_severity(output, retrieved_chunks)
    conf = confidence_score(output)

    # divergence only relevant for summaries
    div = summary_divergence(output, retrieved_chunks)

    hallucinated = sev > 0.35 and g < 0.55

    return {
        "groundedness": round(g, 4),
        "severity": round(sev, 4),
        "confidence": round(conf, 4),
        "summary_divergence": round(div, 4),
        "hallucinated": hallucinated,
        "verdict": "hallucinated" if hallucinated else "grounded",
    }
