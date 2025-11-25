"""
session_rag_search.py
---------------------
Search utilities for a SINGLE session (ONE PDF) RAG artifacts.

Provides:
  - search_session_chunks(session_id, query, top_k)
  - load_session_global_blocks(session_id)
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../../"))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sentence_transformers import SentenceTransformer
import numpy as np

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

Chunk = Dict[str, Any]
Block = Dict[str, Any]


def _get_session_root(session_id: str) -> Path:
    """
    Resolve .../data/local_pipeline/sessions/<session_id>.
    """
    sessions_dir = Path(PROJECT_ROOT) / "data" / "local_pipeline" / "sessions"
    session_root = sessions_dir / session_id
    if not session_root.exists():
        raise FileNotFoundError(f"Session root not found: {session_root}")
    return session_root


def _load_rag_paths(session_id: str) -> Dict[str, Path]:
    session_root = _get_session_root(session_id)
    rag_dir = session_root / "rag"
    if not rag_dir.exists():
        raise FileNotFoundError(f"RAG dir not found for session {session_id}: {rag_dir}")

    chunks_path = rag_dir / "chunks.json"
    blocks_path = rag_dir / "blocks.json"
    emb_path = rag_dir / "chunk_embeddings.npy"

    if not chunks_path.exists() or not emb_path.exists():
        raise FileNotFoundError(
            f"Missing RAG artifacts for session {session_id} in {rag_dir}"
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

    Args:
        session_id: which session to search (folder name).
        query: user query text.
        top_k: number of chunks to return.

    Returns:
        List of dicts:
          - chunk (full chunk dict)
          - score (similarity score)
    """
    paths = _load_rag_paths(session_id)

    with open(paths["chunks"], "r", encoding="utf-8") as f:
        chunks: List[Chunk] = json.load(f)

    chunk_embeddings = np.load(paths["embeddings"])
    if chunk_embeddings.shape[0] != len(chunks):
        raise ValueError(
            f"Embeddings count ({chunk_embeddings.shape[0]}) "
            f"!= chunks count ({len(chunks)}) for session {session_id}"
        )

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
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
    for idx in top_idx:
        idx = int(idx)
        results.append(
            {
                "chunk": chunks[idx],
                "score": float(sims[idx]),
            }
        )

    return results


def load_session_global_blocks(session_id: str) -> List[Block]:
    """
    Load precomputed global blocks for whole-document tasks.
    """
    paths = _load_rag_paths(session_id)
    blocks_path = paths["blocks"]

    if not blocks_path.exists():
        # It's okay if there are no blocks, just return empty list
        return []

    with open(blocks_path, "r", encoding="utf-8") as f:
        blocks: List[Block] = json.load(f)

    return blocks


if __name__ == "__main__":
    # Quick manual test using the MOST RECENT session (like your span_adapter main)
    sessions_dir = Path(PROJECT_ROOT) / "data" / "local_pipeline" / "sessions"
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
            f"- score={r['score']:.3f}, pages {c['page_start']}–{c['page_end']}, len={c['char_len']}"
        )
        print("  ", c["text"][:200].replace('\n', ' '), "...\n")

    blocks = load_session_global_blocks(test_session_id)
    print(f"Global blocks: {len(blocks)}")
