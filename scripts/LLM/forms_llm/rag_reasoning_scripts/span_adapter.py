"""
span_adapter.py
----------------
Span-level document utilities for a SINGLE normalized JSON file.

This module is purely about structure:
  normalized JSON  -> spans -> sorted spans -> local chunks -> global blocks.

No embeddings, no LLM, no FAISS. Just text + metadata.

Preferred text field order:
  1. text_display
  2. text_preserved
  3. text
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

# ---------------------------------------------------------------------
# 🔧 Ensure project root on sys.path (same pattern as other scripts)
# ---------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))          # .../scripts/rag_reasoning_scripts
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../../"))  # .../doc-understand

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

Span = Dict[str, Any]
Chunk = Dict[str, Any]
Block = Dict[str, Any]


# ---------------------------------------------------------------------
# 1) Load spans from ONE normalized JSON file
# ---------------------------------------------------------------------
def load_spans_from_normalized(normalized_path: str) -> List[Span]:
    """
    Load spans from a single normalized JSON file produced by the pipeline.

    Args:
        normalized_path: path to '.../normalized/<file>_normalized.json'

    Returns:
        List[Span] where each span has at least:
          - doc_id
          - page
          - span_id
          - text        (chosen from text_display / text_preserved / text)
          - bbox {Top, Left, Width, Height}
          - conf
    """
    path = Path(normalized_path)
    if not path.exists():
        raise FileNotFoundError(f"Normalized file not found: {normalized_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected list of spans in {normalized_path}, got {type(data)}")

    spans: List[Span] = []
    for idx, span in enumerate(data):
        page = int(span.get("page", 1))

        bbox = span.get("bbox") or span.get("bounding_box") or {}
        norm_bbox = {
            "Top": float(bbox.get("Top", bbox.get("y", 0.0))),
            "Left": float(bbox.get("Left", bbox.get("x", 0.0))),
            "Width": float(bbox.get("Width", bbox.get("w", 0.0))),
            "Height": float(bbox.get("Height", bbox.get("h", 0.0))),
        }

        # Preferred text selection: text_display > text_preserved > text
        text_value = (
            span.get("text_display")
            or span.get("text_preserved")
            or span.get("text")
            or ""
        )

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
        raise ValueError(f"No spans loaded from {normalized_path}")

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

    Args:
        spans: list of spans already sorted in reading order.
        target_char_len: preferred approximate length of each chunk.
        max_char_len: hard cap; if total chars exceed this, we flush even
                      if we are mid-sentence.

    Returns:
        List[Chunk]:
          - chunk_id
          - text
          - char_len
          - page_start
          - page_end
          - span_ids
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
            # Recompute for a fresh chunk
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

    Returns:
        List[Block]:
          - block_id
          - text
          - char_len
          - page_start
          - page_end
          - span_ids
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
    return blocks


# ---------------------------------------------------------------------
# 🔍 Quick manual test (single-PDF, latest session)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # 1) Find sessions directory
    sessions_dir = Path(PROJECT_ROOT) / "data" / "local_pipeline" / "sessions"

    if not sessions_dir.exists():
        print(f"Sessions folder not found: {sessions_dir}")
        sys.exit(1)

    # 2) Get all session_* folders and pick the most recent one
    session_dirs = [
        d for d in sessions_dir.iterdir()
        if d.is_dir() and d.name.startswith("session_")
    ]
    if not session_dirs:
        print(f"No session folders found in {sessions_dir}")
        sys.exit(1)

    latest_session = max(session_dirs, key=lambda d: d.stat().st_mtime)
    normalized_dir = latest_session / "normalized"

    if not normalized_dir.exists():
        print(f"No 'normalized' folder found in {latest_session}")
        sys.exit(1)

    # 3) Get first JSON file inside normalized/
    json_files = list(normalized_dir.glob("*.json"))
    if not json_files:
        print(f"No normalized JSON files found in {normalized_dir}")
        sys.exit(1)

    example_normalized = json_files[0]
    print(f"Using normalized file:\n  {example_normalized}\n")

    # 4) Run pipeline: spans -> sorted spans -> chunks -> blocks
    spans = load_spans_from_normalized(str(example_normalized))
    print(f"Loaded spans: {len(spans)}")

    sorted_spans = sort_spans_reading_order(spans)
    chunks = make_local_chunks(sorted_spans)
    blocks = make_global_blocks(sorted_spans)

    print(f"Local chunks: {len(chunks)}")
    print(f"Global blocks: {len(blocks)}")

    # 5) Show first few LOCAL chunks in the terminal
    print("\n=== SAMPLE LOCAL CHUNKS (sentence-aware, page-aligned) ===")
    for i, chunk in enumerate(chunks[:5]):  # change 5 to 10 if you want more
        print("\n-----------------------------")
        print(
            f"Chunk {i} | chars={chunk['char_len']} | "
            f"pages {chunk['page_start']}–{chunk['page_end']} "
            f"| span_ids={chunk['span_ids'][0]}–{chunk['span_ids'][-1]}"
        )
        print(chunk["text"])

    # 6) Show first few GLOBAL blocks in the terminal
    print("\n=== SAMPLE GLOBAL BLOCKS (whole-document view, sentence-aware) ===")
    for i, block in enumerate(blocks[:3]):  # show first 3 blocks
        print("\n=============================")
        print(
            f"Block {i} | chars={block['char_len']} | "
            f"pages {block['page_start']}–{block['page_end']} "
            f"| span_ids={block['span_ids'][0]}–{block['span_ids'][-1]}"
        )
        print(block["text"])  # full block text (no truncation)

    # 7) Save all chunks and blocks to JSON for inspection in VS Code
    debug_dir = latest_session / "rag_debug"
    debug_dir.mkdir(exist_ok=True)

    chunks_path = debug_dir / "local_chunks.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    blocks_path = debug_dir / "global_blocks.json"
    with open(blocks_path, "w", encoding="utf-8") as f:
        json.dump(blocks, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(chunks)} chunks to: {chunks_path}")
    print(f"Saved {len(blocks)} blocks to: {blocks_path}")
