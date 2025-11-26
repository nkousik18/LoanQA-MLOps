"""
faiss_from_rag_builder.py
-------------------------
Build a FAISS index for a SINGLE session (ONE PDF),
REUSING the embeddings from the existing numpy-based RAG.

Flow:
  1) Run numpy builder first:
       build_session_rag_from_normalized(normalized_path)
     -> writes:
          sessions/<session_id>/rag/chunks.json
          sessions/<session_id>/rag/chunk_embeddings.npy

  2) Then run:
       build_faiss_index_from_rag(session_id)

  3) This builds a FAISS IndexFlatIP over the SAME embeddings and
     saves it as:
       sessions/<session_id>/rag_faiss/faiss_index.bin
       sessions/<session_id>/rag_faiss/chunks.json  (copy of source)
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../../"))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np

try:
    import faiss
except ImportError as e:
    raise ImportError(
        "faiss (or faiss-cpu) is not installed. "
        "Install with: pip install faiss-cpu"
    ) from e


def _get_session_root(session_id: str) -> Path:
    sessions_dir = Path(PROJECT_ROOT) / "data" / "local_pipeline" / "sessions"
    session_root = sessions_dir / session_id
    if not session_root.exists():
        raise FileNotFoundError(f"Session root not found: {session_root}")
    return session_root


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors / norms


def build_faiss_index_from_rag(session_id: str) -> Dict[str, Any]:
    """
    Build FAISS index for a SINGLE session, using existing numpy RAG artifacts.

    Reads from:
      sessions/<session_id>/rag/chunks.json
      sessions/<session_id>/rag/chunk_embeddings.npy

    Writes into:
      sessions/<session_id>/rag_faiss/chunks.json
      sessions/<session_id>/rag_faiss/faiss_index.bin
    """
    session_root = _get_session_root(session_id)

    rag_dir = session_root / "rag"
    rag_faiss_dir = session_root / "rag_faiss"
    rag_faiss_dir.mkdir(exist_ok=True)

    src_chunks_path = rag_dir / "chunks.json"
    src_emb_path = rag_dir / "chunk_embeddings.npy"

    if not src_chunks_path.exists() or not src_emb_path.exists():
        raise FileNotFoundError(
            f"Expected numpy RAG artifacts not found for session {session_id} in {rag_dir}"
        )

    # Load embeddings and normalize them
    embeddings = np.load(src_emb_path)
    embeddings = _l2_normalize(embeddings)
    num_chunks, dim = embeddings.shape

    # Build FAISS index (cosine via inner product on normalized vectors)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # ---- Save FAISS artifacts under rag_faiss/ ----
    dst_chunks_path = rag_faiss_dir / "chunks.json"
    dst_index_path = rag_faiss_dir / "faiss_index.bin"

    # Copy chunks.json content into rag_faiss
    with open(src_chunks_path, "r", encoding="utf-8") as f_src, \
         open(dst_chunks_path, "w", encoding="utf-8") as f_dst:
        chunks = json.load(f_src)
        json.dump(chunks, f_dst, ensure_ascii=False, indent=2)

    faiss.write_index(index, str(dst_index_path))

    return {
        "session_id": session_id,
        "session_root": str(session_root),
        "rag_dir": str(rag_dir),
        "rag_faiss_dir": str(rag_faiss_dir),
        "num_chunks": num_chunks,
        "embedding_dim": dim,
        "faiss_chunks_path": str(dst_chunks_path),
        "faiss_index_path": str(dst_index_path),
    }


if __name__ == "__main__":
    # 🔍 Test: use the latest session that already has rag/chunks.json + chunk_embeddings.npy
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

    info = build_faiss_index_from_rag(test_session_id)
    print(json.dumps(info, indent=2))
