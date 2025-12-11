# scripts/LLM/forms_llm/planner/doc_task_answer_llm.py

from __future__ import annotations

import os
import sys
from functools import lru_cache
from typing import Dict, Any, List

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.LLM.forms_llm.planner.doc_tasks import (
    DocTask,
    DocTaskKind,
    DocTaskContext,
)
from scripts.LLM.forms_llm.llm_clients.groq_client import call_groq_chat

# Reuse central logger infra from aws_extraction_scripts
from scripts.aws_extraction_scripts.log_utils import get_logger

# ✅ prompts live here: scripts/LLM/prompts_form/
PROMPTS_DIR = os.path.join(PROJECT_ROOT, "scripts", "LLM", "prompts_form")

LOGGER = get_logger(__name__)


# ---------------------------------------------------------
# Helpers: load prompt text, persona, language
# ---------------------------------------------------------
@lru_cache(maxsize=16)
def _load_prompt_file(name: str) -> str:
    """
    Load a prompt template from scripts/LLM/prompts_form/<name>.
    Cached so we don't keep hitting disk.
    """
    path = os.path.join(PROMPTS_DIR, name)
    if not os.path.exists(path):
        LOGGER.error(f"[doc_task_answer_llm] Prompt file not found: {path}")
        raise FileNotFoundError(f"Prompt file not found: {path}")

    LOGGER.debug(f"[doc_task_answer_llm] Loading prompt file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _language_instruction(language: str) -> str:
    """
    Small language hint appended to the system prompt,
    on top of what the prompt file already says.
    """
    lang = (language or "").lower()
    if lang in {"", "en", "english"}:
        return "Respond in clear, concise English."
    if lang in {"te", "telugu"}:
        return (
            "Respond primarily in natural Telugu using Telugu script. "
            "Keep all numbers, dates, interest rates, and amounts exactly as given."
        )
    if lang in {"hi", "hindi"}:
        return (
            "Respond primarily in natural Hindi using Devanagari script. "
            "Keep all numbers, dates, interest rates, and amounts exactly as given."
        )
    # generic fallback
    return (
        f"Respond primarily in {language}. "
        "Preserve all numbers, dates, interest rates, and amounts exactly as given."
    )


def _persona_prefix(tone: str | None) -> str:
    """
    Persona snippet based on task.tone.
    """
    if tone == "expert":
        return (
            "Reader persona: finance/legal professional. "
            "You may use more technical language and assume they know basic terms.\n\n"
        )
    # default → borrower_friendly / normal user
    return (
        "Reader persona: everyday user/borrower. "
        "Use simple, non-technical language and avoid heavy jargon.\n\n"
    )


# ---------------------------------------------------------
# System prompts built from external prompt files
# ---------------------------------------------------------
def _system_prompt_for_doc_qa(language: str, tone: str | None) -> str:
    base = _load_prompt_file("qa_prompt.txt")
    return (
        _persona_prefix(tone)
        + base
        + "\n\n"
        + _language_instruction(language)
    )


def _system_prompt_for_doc_explain(language: str, tone: str | None) -> str:
    base = _load_prompt_file("explain_prompt.txt")
    return (
        _persona_prefix(tone)
        + base
        + "\n\n"
        + _language_instruction(language)
    )


def _system_prompt_for_doc_summary(language: str) -> str:
    base = _load_prompt_file("summary_prompt.txt")
    # summary is already user-friendly; no tone switching for now
    return base + "\n\n" + _language_instruction(language)


def _system_prompt_for_doc_translate(language: str) -> str:
    base = _load_prompt_file("translate_prompt.txt")
    # translate prompt already talks about target language; we still append a small hint
    return base + "\n\n" + _language_instruction(language)


# ---------------------------------------------------------
# Core: answer one or many doc_* tasks
# ---------------------------------------------------------
def answer_single_doc_task(
    task: DocTask,
    ctx: DocTaskContext,
) -> Dict[str, Any]:
    """
    Use Groq LLM to answer ONE doc_* task given its DocTaskContext.
    The context already contains the retrieved text (chunks/blocks).
    """
    LOGGER.info(
        f"[doc_task_answer_llm] Answering task_id={task.id} "
        f"kind={task.kind.value} language={task.language} tone={task.tone} "
        f"scope={ctx.scope.value} pages={ctx.pages}"
    )

    if task.kind == DocTaskKind.DOC_QA:
        system_prompt = _system_prompt_for_doc_qa(task.language, task.tone)
        user_prompt = (
            f"User question:\n{task.query}\n\n"
            "<DOCUMENT_CONTEXT>:\n"
            f"{ctx.context_text}"
        )

    elif task.kind == DocTaskKind.DOC_EXPLAIN:
        system_prompt = _system_prompt_for_doc_explain(task.language, task.tone)
        user_prompt = (
            f"Explanation request:\n{task.query}\n\n"
            "<DOCUMENT_CONTEXT>:\n"
            f"{ctx.context_text}"
        )

    elif task.kind == DocTaskKind.DOC_SUMMARY:
        system_prompt = _system_prompt_for_doc_summary(task.language)
        instruction = task.query or (
            "Provide a concise summary of the key points in this document."
        )
        user_prompt = (
            f"Summarisation request:\n{instruction}\n\n"
            "<DOCUMENT_CONTEXT>:\n"
            f"{ctx.context_text}"
        )

    elif task.kind == DocTaskKind.DOC_TRANSLATE:
        system_prompt = _system_prompt_for_doc_translate(task.language)
        instruction = task.query or "Translate the following clauses into the target language."
        user_prompt = (
            f"Translation request:\n{instruction}\n\n"
            "<DOCUMENT_CONTEXT>:\n"
            f"{ctx.context_text}"
        )

    else:
        LOGGER.error(f"[doc_task_answer_llm] Unsupported DocTaskKind for answer: {task.kind}")
        raise ValueError(f"Unsupported DocTaskKind for answer: {task.kind}")

    LOGGER.debug(
        f"[doc_task_answer_llm] Calling Groq LLM for task_id={task.id} "
        f"(context_length={len(ctx.context_text)})"
    )
    answer_text = call_groq_chat(system_prompt, user_prompt)

    LOGGER.info(
        f"[doc_task_answer_llm] Completed task_id={task.id} "
        f"kind={task.kind.value} (answer_length={len(answer_text)})"
    )

    return {
        "task_id": task.id,
        "kind": task.kind.value,
        "scope": ctx.scope.value,
        "language": task.language,
        "tone": task.tone,
        "question_or_instruction": task.query,
        "answer": answer_text,
        "pages": ctx.pages,
        "segments_meta": ctx.segments_meta,
    }


def answer_many_doc_tasks(
    tasks: List[DocTask],
    contexts: Dict[str, DocTaskContext],
) -> Dict[str, Dict[str, Any]]:
    LOGGER.info(
        f"[doc_task_answer_llm] Answering many tasks: count={len(tasks)}"
    )
    results: Dict[str, Dict[str, Any]] = {}
    for task in tasks:
        ctx = contexts.get(task.id)
        if ctx is None:
            LOGGER.error(f"[doc_task_answer_llm] No DocTaskContext found for task_id={task.id}")
            raise KeyError(f"No DocTaskContext found for task_id={task.id}")
        results[task.id] = answer_single_doc_task(task, ctx)
    return results


# ---------------------------------------------------------
# Manual test (same interface as before)
# ---------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path
    from scripts.LLM.forms_llm.planner.doc_task_executor import run_doc_tasks_on_document
    from scripts.LLM.forms_llm.planner.doc_tasks import DocTaskKind, DocScope, DocTask

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

    tasks: List[DocTask] = [
        DocTask(
            id="t1",
            kind=DocTaskKind.DOC_QA,
            query="What is the interest rate and repayment period in this loan?",
            scope=DocScope.LOCAL,
            language="en",
            tone="borrower_friendly",
        ),
        DocTask(
            id="t2",
            kind=DocTaskKind.DOC_EXPLAIN,
            query="Explain the borrower repayment obligations.",
            scope=DocScope.LOCAL,
            language="en",
            tone="borrower_friendly",
        ),
        DocTask(
            id="t3",
            kind=DocTaskKind.DOC_SUMMARY,
            query="Summarise the key financial terms of this agreement.",
            scope=DocScope.GLOBAL,
            language="en",
        ),
        DocTask(
            id="t4",
            kind=DocTaskKind.DOC_TRANSLATE,
            query="Translate the key clauses into Telugu.",
            scope=DocScope.LOCAL,
            language="te",
        ),
        DocTask(
            id="t5",
            kind=DocTaskKind.DOC_QA,
            query="What is the interest rate stated in this agreement?",
            scope=DocScope.LOCAL,
            language="en",
            tone="borrower_friendly",
        ),
    ]

    ctx_dict = run_doc_tasks_on_document(
        tasks=tasks,
        session_id=test_session_id,
        top_k_local=8,
    )

    answers = answer_many_doc_tasks(tasks, ctx_dict)

    for task_id, result in answers.items():
        print("\n==============================")
        print(f"Task {task_id} | kind={result['kind']} | language={result['language']}")
        print("Pages used:", result["pages"])
        print("\nAnswer:\n", result["answer"])
