# scripts/planner/doc_tasks.py

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any


class DocTaskKind(str, Enum):
    DOC_QA = "doc_qa"
    DOC_EXPLAIN = "doc_explain"
    DOC_SUMMARY = "doc_summary"
    DOC_TRANSLATE = "doc_translate"      # <-- ⭐ NEW


class DocScope(str, Enum):
    LOCAL = "local"     # use top-k chunks
    GLOBAL = "global"   # use global blocks


@dataclass
class DocTask:
    """
    ONE document task in the plan.

    Examples:
      QA:
        DocTask(id="t1", kind=DOC_QA, query="What is interest rate?", scope=LOCAL)

      Translate:
        DocTask(id="t4", kind=DOC_TRANSLATE, query="", scope=GLOBAL, language="es")
    """
    id: str
    kind: DocTaskKind

    # For QA / explain / summary → query is required
    # For translation → query can be empty (we ignore it)
    query: str

    scope: DocScope

    # OUTPUT language (target language for translation)
    language: str = "en"

    # For explanation / QA: tone guidance
    tone: str | None = None


@dataclass
class DocTaskContext:
    """
    Result of running the document-layer part for ONE doc task.
    (No LLM yet, just text + metadata.)
    """
    task_id: str
    kind: DocTaskKind
    scope: DocScope
    language: str
    tone: str | None

    # Retrieval output
    context_text: str
    segments_meta: List[Dict[str, Any]]
    pages: List[int] = field(default_factory=list)
