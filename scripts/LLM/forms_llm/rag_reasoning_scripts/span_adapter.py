"""
span_adapter.py
----------------
Span-level document utilities for a SINGLE SEGMENTED JSON file.

This module is purely about structure:
  segmented JSON  -> spans -> sorted spans -> local chunks -> global blocks

No embeddings, no LLM, no FAISS. Just text + metadata.

Source:
  - Output of segment_text.py
  - Each span has: doc_id, page, span_id, text, conf, bbox
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

# ---------------------------------------------------------------------
# 🔧 Ensure project root on sys.path (same pattern as other scripts)
# ---------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))                   # .../scripts/LLM/forms_llm/rag_reasoning_scripts
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../../"))  # .../doc-understand

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------
# 📦 Imports from OCR config + GCS utils
# ---------------------------------------------------------------------
from google.cloud import storage

from scripts.aws_extraction_scripts.config import (
    LIVE_SESSIONS_DIR,
    USE_GCS_OUTPUT,
    GCS_BUCKET,
    to_gcs_key,
)
from scripts.aws_extraction_scripts.gcs_utils import read_json, logical_exists
from scripts.aws_extraction_scripts.log_utils import get_logger

LOGGER = get_logger(__name__)

Span = Dict[str, Any]
Chunk = Dict[str, Any]
Block = Dict[str, Any]


# ---------------------------------------------------------------------
# 1) Load spans from ONE SEGMENTED JSON file (GCS-aware)
# ---------------------------------------------------------------------
def load_spans_from_segmented(segmented_path: str) -> List[Span]:
    """
    Load spans from a single SEGMENTED JSON file produced by segment_text.py.

    Storage is GCS-aware:
      - When USE_GCS_OUTPUT=True, this uses gcs_utils.read_json(), so the
        file can live only in GCS (no local copy).
      - When USE_GCS_OUTPUT=False, it falls back to local filesystem.

    Args:
        segmented_path: path to '.../segmented/<file>_segmented.json'
                        (absolute or under PROJECT_ROOT)

    Returns:
        List[Span] where each span has at least:
          - doc_id
          - page
          - span_id
          - text
          - bbox {Top, Left, Width, Height}
          - conf
    """
    path = Path(segmented_path)
    LOGGER.info("Loading spans from segmented JSON: %s", path)

    # Check logical existence (GCS or local)
    if not logical_exists(path):
        msg = f"Segmented file not found (GCS/local): {segmented_path}"
        LOGGER.error(msg)
        raise FileNotFoundError(msg)

    # Read JSON from GCS or local via helper
    data = read_json(path)

    if not isinstance(data, list):
        msg = f"Expected list of spans in {segmented_path}, got {type(data)}"
        LOGGER.error(msg)
        raise ValueError(msg)

    spans: List[Span] = []
    for idx, span in enumerate(data):
        page = int(span.get("page", 1))

        # segment_text.py writes bbox in Textract-style {Top, Left, Width, Height}
        bbox = span.get("bbox") or {}
        norm_bbox = {
            "Top": float(bbox.get("Top", bbox.get("y", 0.0))),
            "Left": float(bbox.get("Left", bbox.get("x", 0.0))),
            "Width": float(bbox.get("Width", bbox.get("w", 0.0))),
            "Height": float(bbox.get("Height", bbox.get("h", 0.0))),
        }

        text_value = span.get("text") or ""

        spans.append(
            {
                "doc_id": span.get("doc_id"),
                "page": page,
                "span_id": int(span.get("span_id", idx + 1)),
                "text": text_value,
                "bbox": norm_bbox,
                "conf": float(span.get("conf", span.get("confidence", 0.0))),
            }
        )

    if not spans:
        msg = f"No spans loaded from {segmented_path}"
        LOGGER.error(msg)
        raise ValueError(msg)

    LOGGER.info("Loaded %d spans from %s", len(spans), path)
    return spans


# ---------------------------------------------------------------------
# 2) Sort spans into global reading order
# ---------------------------------------------------------------------
def sort_spans_reading_order(spans: List[Span]) -> List[Span]:
    """
    Sort spans into a stable reading order.

    Priority:
      1) page
      2) span_id          (your pipeline's original order)
      3) bbox (Top, Left) as a fallback only
    """
    if not spans:
        return []

    all_have_span_id = all("span_id" in s for s in spans)

    if all_have_span_id:
        return sorted(spans, key=lambda s: (int(s.get("page", 1)), int(s["span_id"])))
    else:
        return sorted(
            spans,
            key=lambda s: (
                int(s.get("page", 1)),
                float(s["bbox"].get("Top", 0.0)),
                float(s["bbox"].get("Left", 0.0)),
            ),
        )


# ---------------------------------------------------------------------
# 3) Build mid-sized chunks for local RAG (clause-level, sentence-aware)
# ---------------------------------------------------------------------
def make_local_chunks(
    spans: List[Span],
    target_char_len: int = 600,
    max_char_len: int = 900,
) -> List[Chunk]:
    """
    Merge spans into mid-sized chunks for local semantic search.

    Alignment rules:
      - spans are assumed to already be in reading order
      - chunks NEVER mix pages: when page changes, we flush
      - chunk boundaries PREFER sentence endings (., ?, !, :, ;)
      - we only break mid-sentence if we are forced by max_char_len
    """
    if not spans:
        return []

    SENTENCE_ENDING = (".", "?", "!", ":", ";")

    chunks: List[Chunk] = []
    current_text: List[str] = []
    current_span_ids: List[int] = []
    current_pages: List[int] = []
    current_len: int = 0  # track char length without recomputing each time

    def flush_chunk():
        nonlocal current_text, current_span_ids, current_pages, current_len
        if not current_text:
            return
        text = " ".join(current_text).strip()
        if not text:
            return
        pages_sorted = sorted(set(current_pages)) or [1]
        chunk = {
            "chunk_id": len(chunks),
            "text": text,
            "char_len": len(text),
            "page_start": pages_sorted[0],
            "page_end": pages_sorted[-1],
            "span_ids": list(current_span_ids),
        }
        chunks.append(chunk)
        current_text = []
        current_span_ids = []
        current_pages = []
        current_len = 0

    last_page = None

    for span in spans:
        span_text = (span.get("text") or "").strip()
        if not span_text:
            continue

        page = int(span.get("page", 1))

        # 🔹 Never mix pages inside a chunk
        if last_page is not None and page != last_page:
            flush_chunk()

        # Very long single span -> its own chunk, even if mid-sentence
        if len(span_text) >= max_char_len:
            flush_chunk()
            chunks.append(
                {
                    "chunk_id": len(chunks),
                    "text": span_text,
                    "char_len": len(span_text),
                    "page_start": page,
                    "page_end": page,
                    "span_ids": [span["span_id"]],
                }
            )
            last_page = page
            continue

        # Compute projected length if we add this span
        if current_len == 0:
            projected_len = len(span_text)
        else:
            projected_len = current_len + 1 + len(span_text)  # +1 for space

        # If adding this span would exceed hard max, flush first
        if projected_len > max_char_len and current_text:
            flush_chunk()
            projected_len = len(span_text)

        # Add span to current chunk
        if current_len == 0:
            current_text.append(span_text)
            current_len = len(span_text)
        else:
            current_text.append(span_text)
            current_len += 1 + len(span_text)  # +1 for space

        current_span_ids.append(span["span_id"])
        current_pages.append(page)
        last_page = page

        # After adding this span:
        # If we've reached target length AND this span ends with sentence punctuation,
        # we flush here to respect sentence boundary.
        if (
            current_len >= target_char_len
            and span_text.strip().endswith(SENTENCE_ENDING)
        ):
            flush_chunk()

    # Final flush
    flush_chunk()
    LOGGER.info("Built %d local chunks", len(chunks))
    return chunks


# ---------------------------------------------------------------------
# 4) Build large blocks for global / whole-document view (sentence-aware)
# ---------------------------------------------------------------------
def make_global_blocks(
    spans: List[Span],
    target_char_len: int = 2500,
    max_char_len: int = 4000,
) -> List[Block]:
    """
    Merge spans into large blocks for whole-document tasks.

    Rules:
      - blocks CAN cross pages (global context)
      - spans follow sorted order
      - block boundaries PREFER sentence endings (., ?, !, :, ;)
      - only break mid-sentence when forced by max_char_len
    """
    if not spans:
        return []

    SENTENCE_ENDING = (".", "?", "!", ":", ";")

    blocks: List[Block] = []
    current_text: List[str] = []
    current_span_ids: List[int] = []
    current_pages: List[int] = []
    current_len: int = 0

    def flush_block():
        nonlocal current_text, current_span_ids, current_pages, current_len
        if not current_text:
            return
        text = " ".join(current_text).strip()
        if not text:
            return
        pages_sorted = sorted(set(current_pages)) or [1]
        block = {
            "block_id": len(blocks),
            "text": text,
            "char_len": len(text),
            "page_start": pages_sorted[0],
            "page_end": pages_sorted[-1],
            "span_ids": list(current_span_ids),
        }
        blocks.append(block)
        current_text = []
        current_span_ids = []
        current_pages = []
        current_len = 0

    for span in spans:
        span_text = (span.get("text") or "").strip()
        if not span_text:
            continue

        page = int(span.get("page", 1))

        # Super-long single span becomes its own block
        if len(span_text) >= max_char_len:
            flush_block()
            blocks.append(
                {
                    "block_id": len(blocks),
                    "text": span_text,
                    "char_len": len(span_text),
                    "page_start": page,
                    "page_end": page,
                    "span_ids": [span["span_id"]],
                }
            )
            continue

        # Projected length if we add this span
        if current_len == 0:
            projected_len = len(span_text)
        else:
            projected_len = current_len + 1 + len(span_text)

        # If adding this span would exceed hard max, flush first
        if projected_len > max_char_len and current_text:
            flush_block()
            projected_len = len(span_text)

        # Add span to current block
        if current_len == 0:
            current_text.append(span_text)
            current_len = len(span_text)
        else:
            current_text.append(span_text)
            current_len += 1 + len(span_text)

        current_span_ids.append(span["span_id"])
        current_pages.append(page)

        # Prefer to flush at sentence boundary once target length reached
        if (
            current_len >= target_char_len
            and span_text.strip().endswith(SENTENCE_ENDING)
        ):
            flush_block()

    flush_block()
    LOGGER.info("Built %d global blocks", len(blocks))
    return blocks


# ---------------------------------------------------------------------
# 🔍 Quick manual test (single-PDF, latest session, SEGMENTED – GCS-aware)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # 1) Find sessions directory from central config
    sessions_dir = LIVE_SESSIONS_DIR

    if not sessions_dir.exists():
        LOGGER.error("Sessions folder not found: %s", sessions_dir)
        print(f"Sessions folder not found: {sessions_dir}")
        sys.exit(1)

    # 2) Get all session_* folders and pick the most recent one
    session_dirs = [
        d for d in sessions_dir.iterdir()
        if d.is_dir() and d.name.startswith("session_")
    ]
    if not session_dirs:
        LOGGER.error("No session folders found in %s", sessions_dir)
        print(f"No session folders found in {sessions_dir}")
        sys.exit(1)

    latest_session = max(session_dirs, key=lambda d: d.stat().st_mtime)
    segmented_dir = latest_session / "segmented"

    # 3) Find segmented JSON either in GCS (preferred) or locally
    json_paths: List[Path] = []

    if USE_GCS_OUTPUT:
        client = storage.Client()
        prefix = to_gcs_key(segmented_dir)
        if not prefix.endswith("/"):
            prefix += "/"

        LOGGER.info(
            "[GCS] Looking for segmented JSONs under gs://%s/%s",
            GCS_BUCKET,
            prefix,
        )
        blobs = client.list_blobs(GCS_BUCKET, prefix=prefix)
        for blob in blobs:
            name = blob.name
            if not name.endswith(".json"):
                continue
            rel = name[len(prefix):]
            if not rel or rel.endswith("/"):
                continue
            json_paths.append(segmented_dir / rel)
    else:
        json_paths = list(segmented_dir.glob("*.json"))

    if not json_paths:
        msg = (
            "No segmented JSON files found for latest session. "
            f"Checked segmented_dir={segmented_dir}"
        )
        LOGGER.error(msg)
        print(msg)
        sys.exit(1)

    example_segmented = json_paths[0]
    LOGGER.info("Using segmented file (logical path): %s", example_segmented)
    print(f"Using segmented file (logical path):\n  {example_segmented}\n")

    # 4) Run pipeline: spans -> sorted spans -> chunks -> blocks
    spans = load_spans_from_segmented(str(example_segmented))
    LOGGER.info("Loaded %d spans in manual test", len(spans))
    print(f"Loaded spans: {len(spans)}")

    sorted_spans = sort_spans_reading_order(spans)
    chunks = make_local_chunks(sorted_spans)
    blocks = make_global_blocks(sorted_spans)

    LOGGER.info("Manual test produced %d chunks and %d blocks", len(chunks), len(blocks))
    print(f"Local chunks: {len(chunks)}")
    print(f"Global blocks: {len(blocks)}")

    # 5) Save all chunks and blocks to JSON for inspection (LOCAL debug only)
    debug_dir = latest_session / "rag_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = debug_dir / "local_chunks.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    blocks_path = debug_dir / "global_blocks.json"
    with open(blocks_path, "w", encoding="utf-8") as f:
        json.dump(blocks, f, ensure_ascii=False, indent=2)

    LOGGER.info("Saved %d chunks to %s", len(chunks), chunks_path)
    LOGGER.info("Saved %d blocks to %s", len(blocks), blocks_path)

    print(f"\nSaved {len(chunks)} chunks to: {chunks_path}")
    print(f"Saved {len(blocks)} blocks to: {blocks_path}")
