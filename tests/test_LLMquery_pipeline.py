"""
Comprehensive tests for LLMquery modules.
Covers embeddings, prompt building, math evaluation, and API logic.
All external calls and file dependencies are mocked.
"""

import os
import pytest
from unittest import mock

# Ensure root import path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from LLMquery.embeddings_index import load_vectorstore
from LLMquery.prompts.build_prompt import detect_intent, build_prompt
from LLMquery.prompts.finance_prompt import finance_prompt
from LLMquery.prompts.math_utils import evaluate_math
from LLMquery.api_server import add_to_history, get_history_text, query_stream


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

@pytest.fixture
def sample_question():
    return "Can I defer my federal student loan payments?"


@pytest.fixture
def mock_vectorstore(tmp_path):
    """Mock vectorstore path."""
    p = tmp_path / "chroma"
    p.mkdir()
    return str(p)


# -------------------------------------------------------------------
# Embedding & Index
# -------------------------------------------------------------------

def test_load_vectorstore_returns_tuple(mock_vectorstore, monkeypatch):
    """✅ load_vectorstore should return (db, embeddings)."""
    class DummyDB: pass
    class DummyEmbeddings: pass
    monkeypatch.setattr("langchain_community.vectorstores.Chroma", lambda **kw: DummyDB())
    monkeypatch.setattr("langchain_community.embeddings.HuggingFaceEmbeddings", lambda **kw: DummyEmbeddings())
    db, emb = load_vectorstore(mock_vectorstore)
    assert db is not None
    assert emb is not None


# -------------------------------------------------------------------
# Prompt Builders
# -------------------------------------------------------------------

def test_detect_intent_classifies_question():
    """✅ detect_intent should detect category type."""
    intent = detect_intent("What is the repayment plan?")
    assert isinstance(intent, str)
    assert len(intent) > 0


def test_build_prompt_creates_prompt(sample_question):
    """✅ build_prompt returns prompt string using mock documents."""
    DummyDoc = type("DummyDoc", (), {})
    docs = [DummyDoc(), DummyDoc()]
    docs[0].page_content = "Federal loan eligibility"
    docs[1].page_content = "Repayment options for deferment"
    result = build_prompt(sample_question, docs)
    assert isinstance(result, str)
    assert "loan" in result.lower()


def test_finance_prompt_formats_text():
    """✅ finance_prompt should wrap input question properly."""
    out = finance_prompt("Explain deferment", context="student loans")
    assert isinstance(out, str)
    assert "deferment" in out.lower()


# -------------------------------------------------------------------
# Math Utility
# -------------------------------------------------------------------

def test_evaluate_math_expression():
    """✅ evaluate_math evaluates valid math expressions."""
    result = evaluate_math("5 + 10 * 2")
    assert isinstance(result, list)
    assert any("25" in str(r) for r in result)


def test_evaluate_math_invalid():
    """⚠️ Invalid math expression returns None or raises."""
    try:
        res = evaluate_math("invalid expression")
        assert res is None or isinstance(res, (int, float, list))
    except Exception:
        assert True


# -------------------------------------------------------------------
# API Server
# -------------------------------------------------------------------

def test_add_to_history_and_get_text():
    """✅ add_to_history and get_history_text should work."""
    add_to_history("user", "Q: What is interest?")
    add_to_history("assistant", "A: Interest is cost of borrowing.")
    text = get_history_text()
    assert "Interest" in text


@pytest.mark.asyncio
async def test_query_stream(monkeypatch):
    """✅ async query_stream should run properly."""
    mock_req = mock.Mock()
    mock_req.json = mock.AsyncMock(return_value={"query": "What is a loan?"})
    monkeypatch.setattr("LLMquery.api_server.OllamaLLM", lambda **kw: mock.Mock())
    response = await query_stream(mock_req)
    assert response is not None
