# scripts/LLM/forms_llm/planner/doc_task_executor.py

from __future__ import annotations

import os
import sys
from typing import List, Dict, Any

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.LLM.forms_llm.planner.doc_tasks import (
    DocTask,
    DocTaskKind,
    DocScope,
    DocTaskContext,
)

# ⬇️ numpy-based RAG tools (NO FAISS here)
from scripts.LLM.forms_llm.rag_reasoning_scripts.session_rag_search import (
    search_session_chunks,
    load_session_global_blocks,
)

Chunk = Dict[str, Any]
Block = Dict[str, Any]


# ---------------------------------------------------------
# Low-level helpers (using ONLY numpy search)
# ---------------------------------------------------------
def _get_local_chunks(
    session_id: str,
    query: str,
    top_k: int,
) -> List[Chunk]:
    """
    Local semantic search over chunks using numpy backend.
    """
    results = search_session_chunks(session_id, query, top_k=top_k)
    return [r["chunk"] for r in results]


def _get_global_blocks(session_id: str) -> List[Block]:
    """
    Load pre-built global blocks (whole-document view).
    """
    return load_session_global_blocks(session_id)


def _context_from_chunks(
    chunks: List[Chunk],
) -> tuple[str, List[Dict[str, Any]], List[int]]:
    """
    Turn chunks into a single context text + metadata + page list.
    """
    if not chunks:
        return (
            "No relevant clauses were retrieved from the document.",
            [],
            [],
        )

    parts: List[str] = []
    meta: List[Dict[str, Any]] = []
    pages: List[int] = []

    for idx, c in enumerate(chunks):
        parts.append(
            f"[CLAUSE {idx+1} | pages {c['page_start']}–{c['page_end']}]\n{c['text']}"
        )
        meta.append(
            {
                "clause_id": idx + 1,
                "page_start": c["page_start"],
                "page_end": c["page_end"],
                "span_ids": c["span_ids"],
                "char_len": c["char_len"],
            }
        )
        pages.extend(range(c["page_start"], c["page_end"] + 1))

    pages_unique = sorted(set(pages))
    context_text = "\n\n".join(parts)
    return context_text, meta, pages_unique


def _context_from_blocks(
    blocks: List[Block],
) -> tuple[str, List[Dict[str, Any]], List[int]]:
    """
    Turn blocks into a single context text + metadata + page list.
    """
    if not blocks:
        return (
            "No document content was available for summarization or explanation.",
            [],
            [],
        )

    parts: List[str] = []
    meta: List[Dict[str, Any]] = []
    pages: List[int] = []

    for idx, b in enumerate(blocks):
        parts.append(
            f"[BLOCK {idx+1} | pages {b['page_start']}–{b['page_end']}]\n{b['text']}"
        )
        meta.append(
            {
                "block_id": idx + 1,
                "page_start": b["page_start"],
                "page_end": b["page_end"],
                "span_ids": b["span_ids"],
                "char_len": b["char_len"],
            }
        )
        pages.extend(range(b["page_start"], b["page_end"] + 1))

    pages_unique = sorted(set(pages))
    context_text = "\n\n".join(parts)
    return context_text, meta, pages_unique


# ---------------------------------------------------------
# Core: run ONE doc_* task on the document
# ---------------------------------------------------------
def run_single_doc_task_on_document(
    task: DocTask,
    session_id: str,
    top_k_local: int = 8,
) -> DocTaskContext:
    """
    For each doc_qa / doc_explain / doc_summary / doc_translate task:
        - LOCAL: search chunks (numpy)
        - GLOBAL: use global blocks
    Store retrieved text + metadata (NO LLM).
    """

    # Translation tasks DO NOT use query content, but still follow the same retrieval rules
    if task.scope == DocScope.LOCAL:
        chunks = _get_local_chunks(
            session_id=session_id,
            query=task.query,
            top_k=top_k_local,
        )
        context_text, segments_meta, pages = _context_from_chunks(chunks)
    else:
        blocks = _get_global_blocks(session_id)
        context_text, segments_meta, pages = _context_from_blocks(blocks)

    return DocTaskContext(
        task_id=task.id,
        kind=task.kind,
        scope=task.scope,
        language=task.language,
        tone=task.tone,
        context_text=context_text,
        segments_meta=segments_meta,
        pages=pages,
    )


# ---------------------------------------------------------
# Core: run MANY doc_* tasks
# ---------------------------------------------------------
def run_doc_tasks_on_document(
    tasks: List[DocTask],
    session_id: str,
    top_k_local: int = 8,
) -> Dict[str, DocTaskContext]:
    """
    Run a list of doc_* tasks on ONE session using numpy RAG.
    Returns:
        dict[task_id] -> DocTaskContext
    """
    results: Dict[str, DocTaskContext] = {}

    for task in tasks:
        if task.kind not in {
            DocTaskKind.DOC_QA,
            DocTaskKind.DOC_EXPLAIN,
            DocTaskKind.DOC_SUMMARY,
            DocTaskKind.DOC_TRANSLATE,     # <-- ⭐ translation supported
        }:
            raise ValueError(
                f"run_doc_tasks_on_document only supports doc_* tasks, got {task.kind}"
            )

        ctx = run_single_doc_task_on_document(
            task=task,
            session_id=session_id,
            top_k_local=top_k_local,
        )
        results[task.id] = ctx

    return results


# ---------------------------------------------------------
# Manual Test (NO LLM, only retrieval)
# ---------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path

    sessions_dir = Path(PROJECT_ROOT) / "data" / "local_pipeline" / "sessions"
    session_dirs = [
        d for d in sessions_dir.iterdir()
        if d.is_dir() and d.name.startswith("session_")
    ]
    if not session_dirs:
        print(f"No session_* folders found in {sessions_dir}")
        sys.exit(1)

    latest_session = max(session_dirs, key=lambda d: d.stat().st_mtime)
    test_session_id = latest_session.name

    print(f"Using latest session: {test_session_id}")

    tasks = [
        DocTask(
            id="t1",
            kind=DocTaskKind.DOC_QA,
            query="What is the interest rate and repayment duration?",
            scope=DocScope.LOCAL,
            language="en",
        ),
        DocTask(
            id="t2",
            kind=DocTaskKind.DOC_EXPLAIN,
            query="Explain borrower repayment obligations.",
            scope=DocScope.LOCAL,
            language="en",
            tone="borrower_friendly",
        ),
        DocTask(
            id="t3",
            kind=DocTaskKind.DOC_SUMMARY,
            query="Summarise the key financial terms.",
            scope=DocScope.GLOBAL,
            language="en",
        ),
        DocTask(
            id="t4",
            kind=DocTaskKind.DOC_TRANSLATE,   # NEW
            query="",
            scope=DocScope.GLOBAL,
            language="es",     # target: Spanish
        ),
    ]

    results = run_doc_tasks_on_document(
        tasks=tasks,
        session_id=test_session_id,
        top_k_local=8,
    )

    for task_id, ctx in results.items():
        print("\n==============================")
        print(f"Task {task_id} | kind={ctx.kind.value} | scope={ctx.scope.value}")
        print(f"Language={ctx.language} | tone={ctx.tone}")
        print(f"Pages used: {ctx.pages}")
        print("\nContext sample (first 700 chars):")
        print(ctx.context_text[:700], "...")
