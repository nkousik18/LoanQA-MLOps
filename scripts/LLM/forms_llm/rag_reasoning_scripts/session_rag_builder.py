"""
session_rag_builder.py
----------------------
Build RAG artifacts for a SINGLE session (ONE PDF).

Entry point:
  build_session_rag_from_segmented(segmented_path: str)

Assumes:
  segmented_path looks like:
    .../data/local_pipeline/sessions/<session_id>/segmented/<file>.json

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
from google.cloud import storage

from scripts.LLM.forms_llm.rag_reasoning_scripts.span_adapter import (
    load_spans_from_segmented,
    sort_spans_reading_order,
    make_local_chunks,
    make_global_blocks,
)

# GCS / path config + helpers
from scripts.aws_extraction_scripts.config import (
    LIVE_SESSIONS_DIR,
    USE_GCS_OUTPUT,
    GCS_BUCKET,
    to_gcs_key,
)
from scripts.aws_extraction_scripts.gcs_utils import write_json, upload_local_file

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _infer_session_root_and_id(segmented_path: str) -> Tuple[Path, str]:
    """
    Given:
      .../sessions/<session_id>/segmented/<file>.json

    Returns:
      session_root = .../sessions/<session_id>
      session_id   = "<session_id>"
    """
    p = Path(segmented_path).resolve()
    segmented_dir = p.parent            # .../<session_id>/segmented
    session_root = segmented_dir.parent # .../<session_id>
    session_id = session_root.name
    return session_root, session_id


def build_session_rag_from_segmented(segmented_path: str) -> Dict[str, Any]:
    """
    Build RAG artifacts for a SINGLE PDF, starting from the SEGMENTED file path.

    Steps:
      1. Infer session root + session_id from segmented_path.
      2. Load spans, sort them, build local chunks + global blocks.
      3. Compute embeddings for local chunks.
      4. Save artifacts under <session_root>/rag/.

    Storage behaviour:
      - JSON artifacts (chunks/blocks) are saved via gcs_utils.write_json,
        so they are GCS-aware and optionally mirrored locally.
      - Embeddings .npy is always saved locally, and when USE_GCS_OUTPUT=True
        it is also uploaded to GCS via upload_local_file.
    """
    session_root, session_id = _infer_session_root_and_id(segmented_path)
    rag_dir = session_root / "rag"
    rag_dir.mkdir(parents=True, exist_ok=True)  # local dir (for .npy & debug)

    # 1–3: spans -> sorted -> chunks/blocks (from SEGMENTED text, GCS-aware)
    spans = load_spans_from_segmented(segmented_path)
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

    # JSON artifacts -> GCS-aware write
    write_json(chunks_path, chunks)
    write_json(blocks_path, blocks)

    # Embeddings -> save locally, mirror to GCS if enabled
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, chunk_embeddings)

    if USE_GCS_OUTPUT:
        # logical path = emb_path (under PROJECT_ROOT), upload via helper
        upload_local_file(emb_path, emb_path, content_type="application/octet-stream")

    return {
        "session_id": session_id,
        "session_root": str(session_root),
        "segmented": segmented_path,
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
    # For quick testing: use latest session's SEGMENTED file (GCS-aware)
    sessions_dir = LIVE_SESSIONS_DIR
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
    segmented_dir = latest_session / "segmented"

    json_files: list[Path] = []

    if USE_GCS_OUTPUT:
        client = storage.Client()
        prefix = to_gcs_key(segmented_dir)
        if not prefix.endswith("/"):
            prefix += "/"

        print(f"[GCS] Looking for segmented JSONs under gs://{GCS_BUCKET}/{prefix}")
        blobs = client.list_blobs(GCS_BUCKET, prefix=prefix)

        for blob in blobs:
            name = blob.name
            if not name.endswith(".json"):
                continue
            rel = name[len(prefix):]
            if not rel or rel.endswith("/"):
                continue
            json_files.append(segmented_dir / rel)
    else:
        json_files = list(segmented_dir.glob("*.json"))

    if not json_files:
        print("No segmented JSON files found for latest session.")
        sys.exit(1)

    segmented_path = str(json_files[0])
    print(f"Using segmented file (logical path): {segmented_path}")

    info = build_session_rag_from_segmented(segmented_path)
    print(json.dumps(info, indent=2))
