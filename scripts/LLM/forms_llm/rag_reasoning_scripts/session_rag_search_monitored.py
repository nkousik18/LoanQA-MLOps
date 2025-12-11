"""
session_rag_search.py (WITH CLOUD MONITORING)
----------------------------------------------
Search utilities with RAG quality tracking.

Changes from original:
- Added timing for RAG retrieval
- Tracks retrieval quality scores
- Logs to Cloud Monitoring
"""

import os
import sys
import json
import time  # ✅ ADDED for timing
from pathlib import Path
from functools import lru_cache
from typing import List, Dict, Any

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../../"))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sentence_transformers import SentenceTransformer
import numpy as np

# GCS-aware config + helpers
from scripts.aws_extraction_scripts.config import LIVE_SESSIONS_DIR, USE_GCS_OUTPUT
from scripts.aws_extraction_scripts.gcs_utils import read_json, logical_exists
from scripts.aws_extraction_scripts.log_utils import get_logger

# ✅ NEW: Cloud Monitoring
try:
    from scripts.aws_extraction_scripts.cloud_monitoring import log_rag_retrieval
    MONITORING_ENABLED = True
except ImportError:
    MONITORING_ENABLED = False

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

Chunk = Dict[str, Any]
Block = Dict[str, Any]

LOGGER = get_logger("session_rag_search")


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """
    Lazily load and cache the embedding model once per process.
    This avoids re-loading for every search call (important for Streamlit / API).
    """
    LOGGER.info("Loading SentenceTransformer model: %s", EMBEDDING_MODEL_NAME)
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def _get_session_root(session_id: str) -> Path:
    """
    Resolve .../data/local_pipeline/sessions/<session_id> using central config.
    """
    sessions_dir = LIVE_SESSIONS_DIR
    session_root = sessions_dir / session_id
    if not session_root.exists():
        raise FileNotFoundError(f"Session root not found: {session_root}")
    return session_root


def _load_rag_paths(session_id: str) -> Dict[str, Path]:
    """
    Resolve paths for RAG artifacts of a session.

    Notes:
      - chunks.json / blocks.json may live only in GCS; we check existence
        via logical_exists().
      - chunk_embeddings.npy must exist locally (created by session_rag_builder).
    """
    session_root = _get_session_root(session_id)
    rag_dir = session_root / "rag"
    if not rag_dir.exists():
        raise FileNotFoundError(f"RAG dir not found for session {session_id}: {rag_dir}")

    chunks_path = rag_dir / "chunks.json"
    blocks_path = rag_dir / "blocks.json"
    emb_path = rag_dir / "chunk_embeddings.npy"

    # JSONs: check logical existence (GCS or local)
    if not logical_exists(chunks_path):
        raise FileNotFoundError(
            f"chunks.json not found (GCS/local) for session {session_id}: {chunks_path}"
        )

    if not logical_exists(blocks_path):
        LOGGER.warning(
            "blocks.json not found (GCS/local) for session %s: %s",
            session_id,
            blocks_path,
        )

    # Embeddings: must exist locally
    if not emb_path.exists():
        raise FileNotFoundError(
            f"chunk_embeddings.npy not found locally for session {session_id}: {emb_path}"
        )

    LOGGER.info(
        "Resolved RAG paths for session %s: chunks=%s, blocks=%s, embeddings=%s",
        session_id,
        chunks_path,
        blocks_path,
        emb_path,
    )

    return {
        "session_root": session_root,
        "rag_dir": rag_dir,
        "chunks": chunks_path,
        "blocks": blocks_path,
        "embeddings": emb_path,
    }


def search_session_chunks(
    session_id: str,
    query: str,
    top_k: int = 8,
) -> List[Dict[str, Any]]:
    """
    Semantic search over local chunks for a SINGLE session/PDF.
    
    NOW WITH MONITORING: Tracks RAG quality and logs to Cloud Monitoring.

    Args:
        session_id: which session to search (folder name).
        query: user query text.
        top_k: number of chunks to return.

    Returns:
        List of dicts:
          - chunk (full chunk dict)
          - score (similarity score)
    """
    # ✅ START TIMING
    start_time = time.time()
    
    LOGGER.info(
        "Running chunk search for session=%s, top_k=%d, query=%r",
        session_id,
        top_k,
        query[:120],
    )

    paths = _load_rag_paths(session_id)

    # JSON is loaded from GCS or local via helper
    chunks: List[Chunk] = read_json(paths["chunks"])
    LOGGER.info("Loaded %d chunks for session %s", len(chunks), session_id)

    chunk_embeddings = np.load(paths["embeddings"])
    if chunk_embeddings.shape[0] != len(chunks):
        raise ValueError(
            f"Embeddings count ({chunk_embeddings.shape[0]}) "
            f"!= chunks count ({len(chunks)}) for session {session_id}"
        )

    model = _get_model()
    query_vec = model.encode(
        [query],
        show_progress_bar=False,
        convert_to_numpy=True,
    )[0]

    # cosine similarity
    norms = np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_vec)
    sims = np.dot(chunk_embeddings, query_vec) / (norms + 1e-8)

    top_k = min(top_k, len(chunks))
    top_idx = np.argsort(-sims)[:top_k]

    results: List[Dict[str, Any]] = []
    scores_list: List[float] = []  # ✅ Track scores for monitoring
    
    for idx in top_idx:
        idx = int(idx)
        score = float(sims[idx])
        results.append(
            {
                "chunk": chunks[idx],
                "score": score,
            }
        )
        scores_list.append(score)

    # ✅ CALCULATE METRICS
    duration = time.time() - start_time
    avg_score = sum(scores_list) / len(scores_list) if scores_list else 0.0
    top_score = max(scores_list) if scores_list else 0.0
    
    LOGGER.info(
        "Search complete for session %s. Returned %d chunks. "
        "Avg score: %.3f, Top score: %.3f, Duration: %.2fs",
        session_id,
        len(results),
        avg_score,
        top_score,
        duration,
    )
    
    # ✅ LOG TO CLOUD MONITORING
    if MONITORING_ENABLED:
        try:
            log_rag_retrieval(
                session_id=session_id,
                query=query,
                num_chunks=len(results),
                avg_score=avg_score,
                top_score=top_score,
                duration_seconds=duration,
            )
        except Exception as e:
            LOGGER.warning(f"Failed to log RAG metrics to Cloud Monitoring: {e}")

    return results


def load_session_global_blocks(session_id: str) -> List[Block]:
    """
    Load precomputed global blocks for whole-document tasks.

    Note:
      blocks.json is also loaded via read_json(), so it can live only in GCS
      when USE_GCS_OUTPUT=True. If the file doesn't exist at all, we return [].
    """
    paths = _load_rag_paths(session_id)
    blocks_path = paths["blocks"]

    if not logical_exists(blocks_path):
        LOGGER.warning(
            "No blocks.json found for session %s (logical path: %s). Returning empty list.",
            session_id,
            blocks_path,
        )
        return []

    blocks: List[Block] = read_json(blocks_path)
    LOGGER.info("Loaded %d global blocks for session %s", len(blocks), session_id)
    return blocks


if __name__ == "__main__":
    # Quick manual test using the MOST RECENT session
    sessions_dir = LIVE_SESSIONS_DIR
    if not sessions_dir.exists():
        print(f"No sessions directory found: {sessions_dir}")
        sys.exit(1)

    session_dirs = [
        d for d in sessions_dir.iterdir()
        if d.is_dir() and d.name.startswith("session_")
    ]
    if not session_dirs:
        print(f"No session folders in {sessions_dir}")
        sys.exit(1)

    latest_session = max(session_dirs, key=lambda d: d.stat().st_mtime)
    test_session_id = latest_session.name
    print(f"Using latest session: {test_session_id}")

    q = "What is the interest rate?"
    results = search_session_chunks(test_session_id, q, top_k=5)
    print(f"\nTop-{len(results)} chunks for query: {q}")
    for r in results:
        c = r["chunk"]
        print(
            f"- score={r['score']:.3f}, pages {c['page_start']}–{c['page_end']}, "
            f"len={c['char_len']}"
        )
        print("  ", c["text"][:200].replace('\n', ' '), "...\n")

    blocks = load_session_global_blocks(test_session_id)
    print(f"Global blocks: {len(blocks)}")
