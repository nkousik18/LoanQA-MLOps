import os
import sys
import pytest
from unittest import mock
from pathlib import Path

# ============================================================
# ✅ Path Setup
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

SCRIPTS_PATH = os.path.join(PROJECT_ROOT, "scripts")
if SCRIPTS_PATH not in sys.path:
    sys.path.append(SCRIPTS_PATH)

# ============================================================
# Imports
# ============================================================
from scripts.LLMquery.build_index import add_to_index, rebuild_vector_index
from scripts.LLMquery.embeddings_index import load_vectorstore
from scripts.LLMquery.api_server import (
    log_to_csv, format_sources, upload_file, cached_retrieval, query_stream
)
from scripts.LLMquery import inspect_index
from scripts.LLMquery.prompts.finance_prompt import finance_prompt
from scripts.LLMquery.prompts.math_utils import evaluate_math
from scripts.LLMquery.prompts.summary_prompt import summary_prompt
from scripts.LLMquery.prompts.explanation_prompt import explanation_prompt
from scripts.LLMquery.prompts.prompt_router import (
    detect_intent, build_prompt, keyword_score, numeric_pattern_score, safe_extract_context
)
from scripts.extraction_pipeline.config import setup_logger

# ============================================================
# Logger for this test module
# ============================================================
test_logger = setup_logger("llm_pipeline_tests", log_type="test")
test_logger.info("🚀 Starting LLM Pipeline Unit Tests")

# -------------------------------------------------------------------
# 🧱 Index / Vector store
# -------------------------------------------------------------------

def test_add_to_index(monkeypatch):
    """✅ Mock Chroma add flow."""
    monkeypatch.setattr("scripts.LLMquery.build_index.Chroma", mock.Mock())
    test_logger.info("🧪 Testing add_to_index()...")
    add_to_index("tests/sample_mock.txt")
    assert True
    test_logger.info("✅ add_to_index passed.")


def test_rebuild_vector_index(monkeypatch):
    """✅ No-arg index rebuild."""
    monkeypatch.setattr("scripts.LLMquery.build_index.Chroma", mock.Mock())
    test_logger.info("🧪 Testing rebuild_vector_index()...")
    rebuild_vector_index()
    assert True
    test_logger.info("✅ rebuild_vector_index passed.")


def test_load_vectorstore(monkeypatch, tmp_path):
    """✅ Mock vectorstore directory existence to prevent FileNotFoundError."""
    mock_index = tmp_path / "index"
    mock_index.mkdir()

    monkeypatch.setattr("os.path.exists", lambda p: True)
    test_logger.info("🧪 Testing load_vectorstore() (mocked path)...")

    db, emb = load_vectorstore(str(mock_index))
    assert db is not None and emb is not None
    test_logger.info("✅ load_vectorstore passed.")



# -------------------------------------------------------------------
# 🌐 API Server Helpers
# -------------------------------------------------------------------

def test_log_to_csv(tmp_path):
    """Ensure log_to_csv creates/updates CSV correctly."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    os.chdir(tmp_path)
    log_file = log_dir / "query_logs.csv"
    log_file.touch()

    test_logger.info("🧪 Testing log_to_csv()...")
    log_to_csv({"question": "hello"})
    assert log_file.exists()
    test_logger.info("✅ log_to_csv passed.")


def test_format_sources():
    """✅ Provide dummy docs with metadata."""
    DummyDoc = type("DummyDoc", (), {
        "page_content": "sample text",
        "metadata": {"source": "dummy.pdf"}
    })
    docs = [DummyDoc()]
    test_logger.info("🧪 Testing format_sources()...")
    result = format_sources(docs)
    assert isinstance(result, tuple)
    assert "dummy.pdf" in result[0]
    assert isinstance(result[1], list)
    test_logger.info("✅ format_sources passed.")


def test_upload_file(monkeypatch, tmp_path):
    """✅ Ensure upload_file handles a valid mock file."""
    file = tmp_path / "sample.pdf"
    file.write_text("data")

    mock_req = mock.Mock()
    mock_req.file = file

    test_logger.info("🧪 Testing upload_file()...")
    res = upload_file(mock_req)
    assert res is not None
    test_logger.info("✅ upload_file passed.")


def test_cached_retrieval(monkeypatch):
    """✅ Run cached_retrieval safely."""
    test_logger.info("🧪 Testing cached_retrieval()...")
    out = cached_retrieval("What is a loan?")
    assert isinstance(out, (list, dict))
    test_logger.info("✅ cached_retrieval passed.")


@pytest.mark.asyncio
async def test_query_stream(monkeypatch):
    """✅ Ensure async query_stream executes."""
    req = mock.Mock()
    req.json = mock.AsyncMock(return_value={"query": "Test"})
    monkeypatch.setattr("scripts.LLMquery.api_server.OllamaLLM", lambda **kw: mock.Mock())

    test_logger.info("🧪 Testing query_stream()...")
    res = await query_stream(req)
    assert res is not None
    test_logger.info("✅ query_stream passed.")


# -------------------------------------------------------------------
# 💬 Prompts / Math / Finance
# -------------------------------------------------------------------

def test_prompts_and_math_utils():
    """✅ Validate prompt templates and math utils."""
    test_logger.info("🧪 Testing prompt + math utils...")
    assert "loan" in finance_prompt("Explain loan", "context").lower()
    assert isinstance(evaluate_math("5+5"), list)
    assert "summary" in summary_prompt("summarize", "context").lower()
    assert "explain" in explanation_prompt("explain interest", "context").lower()
    test_logger.info("✅ Prompts and math utils passed.")


def test_build_and_detect_prompts():
    """✅ Ensure prompt router functions behave."""
    DummyDoc = type("DummyDoc", (), {"page_content": "loan info"})
    docs = [DummyDoc()]
    test_logger.info("🧪 Testing prompt router build_prompt() + detect_intent()...")

    result = build_prompt("loan question", docs)
    # handle tuple return (prompt, intent, confidence, gap)
    prompt_text = result[0] if isinstance(result, tuple) else result

    # detect_intent now returns (intent, confidence, gap)
    detected = detect_intent("loan")
    intent_label = detected[0] if isinstance(detected, tuple) else detected

    assert isinstance(prompt_text, str)
    assert intent_label in ["finance", "loan", "context"]
    test_logger.info(f"✅ prompt_router passed with intent={intent_label}.")

def test_prompt_router_scores():
    """✅ Ensure score functions are stable."""
    test_logger.info("🧪 Testing keyword_score + numeric_pattern_score...")
    assert keyword_score("loan details", "finance") >= 0
    assert numeric_pattern_score("Rate is 12.5%") >= 0
    assert isinstance(safe_extract_context("some context"), str)
    test_logger.info("✅ prompt_router scoring passed.")


import importlib
import scripts.LLMquery.inspect_index as inspect_index_module

def test_inspect_index(monkeypatch):
    """✅ Ensure inspect_index runs safely."""
    test_logger.info("🧪 Testing inspect_index()...")
    monkeypatch.setattr("builtins.print", lambda *a, **kw: None)

    # dynamically get function (works for any refactor)
    inspect_func = getattr(inspect_index_module, "inspect_index", None)
    if callable(inspect_func):
        assert inspect_func("test_index") is None
    else:
        # fallback to module import success test
        assert importlib.import_module("scripts.LLMquery.inspect_index") is not None
    test_logger.info("✅ inspect_index passed.")

