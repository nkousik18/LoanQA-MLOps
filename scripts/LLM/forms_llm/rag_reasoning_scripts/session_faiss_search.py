"""
session_faiss_search.py
-----------------------
Search FAISS index for a SINGLE session (ONE PDF),

Uses:
  sessions/<session_id>/rag_faiss/chunks.json
  sessions/<session_id>/rag_faiss/faiss_index.bin
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

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import faiss
except ImportError as e:
    raise ImportError(
        "faiss (or faiss-cpu) is not installed. "
        "Install with: pip install faiss-cpu"
    ) from e

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

Chunk = Dict[str, Any]


def _get_session_root(session_id: str) -> Path:
    sessions_dir = Path(PROJECT_ROOT) / "data" / "local_pipeline" / "sessions"
    session_root = sessions_dir / session_id
    if not session_root.exists():
        raise FileNotFoundError(f"Session root not found: {session_root}")
    return session_root


def _load_faiss_paths(session_id: str) -> Dict[str, Path]:
    session_root = _get_session_root(session_id)
    rag_faiss_dir = session_root / "rag_faiss"
    chunks_path = rag_faiss_dir / "chunks.json"
    index_path = rag_faiss_dir / "faiss_index.bin"

    if not rag_faiss_dir.exists():
        raise FileNotFoundError(
            f"FAISS rag dir not found for session {session_id}: {rag_faiss_dir}"
        )
    if not chunks_path.exists() or not index_path.exists():
        raise FileNotFoundError(
            f"Missing FAISS artifacts (chunks.json / faiss_index.bin) for session {session_id} in {rag_faiss_dir}"
        )

    return {
        "session_root": session_root,
        "rag_faiss_dir": rag_faiss_dir,
        "chunks": chunks_path,
        "faiss_index": index_path,
    }


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors / norms


def faiss_search_session_chunks(
    session_id: str,
    query: str,
    top_k: int = 8,
) -> List[Dict[str, Any]]:
    """
    Search FAISS index for a SINGLE session/PDF.

    Requires that you have already run:
      - numpy RAG builder (rag/)
      - build_faiss_index_from_rag(session_id) (rag_faiss/)

    Returns:
        List of dicts:
          - rank
          - chunk
          - score (cosine similarity)
    """
    paths = _load_faiss_paths(session_id)

    # Load chunks metadata from rag_faiss/
    with open(paths["chunks"], "r", encoding="utf-8") as f:
        chunks: List[Chunk] = json.load(f)

    # Load FAISS index
    index = faiss.read_index(str(paths["faiss_index"]))

    # Encode + normalize query
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    q_vec = model.encode([query], show_progress_bar=False, convert_to_numpy=True)
    q_vec = _l2_normalize(q_vec)  # (1, dim)

    # FAISS search
    top_k = min(top_k, len(chunks))
    scores, indices = index.search(q_vec, top_k)

    scores = scores[0]
    indices = indices[0]

    results: List[Dict[str, Any]] = []
    for rank, (score, idx) in enumerate(zip(scores, indices)):
        idx = int(idx)
        if idx < 0 or idx >= len(chunks):
            continue
        results.append(
            {
                "rank": rank,
                "chunk": chunks[idx],
                "score": float(score),
            }
        )

    return results


if __name__ == "__main__":
    # Use latest session for a quick test
    sessions_dir = Path(PROJECT_ROOT) / "data" / "local_pipeline" / "sessions"
    session_dirs = [
        d for d in sessions_dir.iterdir()
        if d.is_dir() and d.name.startswith("session_")
    ]
    if not session_dirs:
        print(f"No session folders found in {sessions_dir}")
        sys.exit(1)

    latest_session = max(session_dirs, key=lambda d: d.stat().st_mtime)
    test_session_id = latest_session.name
    print(f"Using latest session: {test_session_id}")

    q = "What is the interest rate?"
    results = faiss_search_session_chunks(test_session_id, q, top_k=5)
    print(f"\nTop-{len(results)} FAISS chunks for query: {q}")
    for r in results:
        c = r["chunk"]
        print(
            f"- rank={r['rank']} score={r['score']:.3f}, "
            f"pages {c['page_start']}–{c['page_end']}, len={c['char_len']}"
        )
        print("  ", c['text'][:200].replace('\\n', ' '), "...\n")
