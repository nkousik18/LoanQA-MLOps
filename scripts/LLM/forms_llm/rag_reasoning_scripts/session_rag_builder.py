"""
session_rag_builder.py
----------------------
Build RAG artifacts for a SINGLE session (ONE PDF).

Entry point:
  build_session_rag_from_normalized(normalized_path: str)

Assumes:
  normalized_path looks like:
    .../data/local_pipeline/sessions/<session_id>/normalized/<file>.json

Produces under:
  .../sessions/<session_id>/rag/:
    - chunks.json
    - blocks.json
    - chunk_embeddings.npy
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../../"))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sentence_transformers import SentenceTransformer
import numpy as np

from scripts.LLM.forms_llm.rag_reasoning_scripts.span_adapter import (
    load_spans_from_normalized,
    sort_spans_reading_order,
    make_local_chunks,
    make_global_blocks,
)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _infer_session_root_and_id(normalized_path: str) -> Tuple[Path, str]:
    """
    Given:
      .../sessions/<session_id>/normalized/<file>.json

    Returns:
      session_root = .../sessions/<session_id>
      session_id   = "<session_id>"
    """
    p = Path(normalized_path).resolve()
    normalized_dir = p.parent              # .../<session_id>/normalized
    session_root = normalized_dir.parent   # .../<session_id>
    session_id = session_root.name
    return session_root, session_id


def build_session_rag_from_normalized(normalized_path: str) -> Dict[str, Any]:
    """
    Build RAG artifacts for a SINGLE PDF, starting from the normalized file path.

    Steps:
      1. Infer session root + session_id from normalized_path.
      2. Load spans, sort them, build local chunks + global blocks.
      3. Compute embeddings for local chunks.
      4. Save artifacts under <session_root>/rag/.

    Returns:
      dict with metadata (paths, counts).
    """
    session_root, session_id = _infer_session_root_and_id(normalized_path)
    rag_dir = session_root / "rag"
    rag_dir.mkdir(parents=True, exist_ok=True)

    # 1–3: spans -> sorted -> chunks/blocks
    spans = load_spans_from_normalized(normalized_path)
    spans_sorted = sort_spans_reading_order(spans)
    chunks = make_local_chunks(spans_sorted)
    blocks = make_global_blocks(spans_sorted)

    if not chunks:
        raise ValueError(f"No chunks created for session {session_id}")

    # 4: embeddings for local chunks
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    chunk_texts = [c["text"] for c in chunks]
    chunk_embeddings = model.encode(
        chunk_texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    # 5: save artifacts
    chunks_path = rag_dir / "chunks.json"
    blocks_path = rag_dir / "blocks.json"
    emb_path = rag_dir / "chunk_embeddings.npy"

    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    with open(blocks_path, "w", encoding="utf-8") as f:
        json.dump(blocks, f, ensure_ascii=False, indent=2)

    np.save(emb_path, chunk_embeddings)

    return {
        "session_id": session_id,
        "session_root": str(session_root),
        "normalized": normalized_path,
        "rag_dir": str(rag_dir),
        "num_spans": len(spans),
        "num_chunks": len(chunks),
        "num_blocks": len(blocks),
        "embedding_dim": int(chunk_embeddings.shape[1]),
        "chunks_path": str(chunks_path),
        "blocks_path": str(blocks_path),
        "embeddings_path": str(emb_path),
    }


if __name__ == "__main__":
    # For quick testing: reuse the "latest session" logic from span_adapter
    sessions_dir = Path(PROJECT_ROOT) / "data" / "local_pipeline" / "sessions"
    if not sessions_dir.exists():
        print(f"Sessions folder not found: {sessions_dir}")
        sys.exit(1)

    session_dirs = [
        d for d in sessions_dir.iterdir()
        if d.is_dir() and d.name.startswith("session_")
    ]
    if not session_dirs:
        print(f"No session folders found in {sessions_dir}")
        sys.exit(1)

    latest_session = max(session_dirs, key=lambda d: d.stat().st_mtime)
    normalized_dir = latest_session / "normalized"
    json_files = list(normalized_dir.glob("*.json"))
    if not json_files:
        print(f"No normalized JSON files in {normalized_dir}")
        sys.exit(1)

    normalized_path = str(json_files[0])
    print(f"Using normalized file: {normalized_path}")

    info = build_session_rag_from_normalized(normalized_path)
    print(json.dumps(info, indent=2))
