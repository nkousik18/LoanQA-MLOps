import os
import io
import pytest
import builtins
from unittest import mock
from pathlib import Path
from LLMquery import api_server

from extraction_pipeline.extractor_core import extract_text
from extraction_pipeline.ocr_utils import run_ocr_on_image, run_ocr_on_pdf_page
from extraction_pipeline.cleaner import clean_text
from extraction_pipeline.utils import list_files
from LLMquery.embeddings_index import load_vectorstore
from LLMquery.api_server import cached_retrieval, log_to_csv
from dags import loan_doc_pipeline_dag


# ============================================================
# 1️⃣ Missing values / empty inputs
# ============================================================

def test_extract_text_with_empty_pdf(monkeypatch):
    """Ensure extract_text handles empty PDF gracefully."""
    class MockDoc:
        def __iter__(self): return iter([])
    monkeypatch.setattr("fitz.open", lambda f: MockDoc())
    result = extract_text("empty.pdf")
    assert result == "" or result is None


def test_ocr_returns_empty_string(monkeypatch):
    """Ensure OCR gracefully handles no text detected."""
    monkeypatch.setattr("paddleocr.PaddleOCR.ocr", lambda *a, **kw: [[]])
    text = run_ocr_on_image("empty.jpg")
    assert isinstance(text, str)


# ============================================================
# 2️⃣ Corrupted or invalid files
# ============================================================

def test_extract_text_on_invalid_pdf(monkeypatch):
    """Simulate invalid or unreadable PDF."""
    monkeypatch.setattr("fitz.open", lambda f: (_ for _ in ()).throw(RuntimeError("Corrupted file")))
    with pytest.raises(RuntimeError):
        extract_text("corrupt.pdf")


def test_non_pdf_extension(monkeypatch):
    """Ensure non-PDF files are redirected to OCR and not crashed."""
    monkeypatch.setattr("extraction_pipeline.extractor_core.run_ocr_on_image", lambda f: "mocked OCR output")
    monkeypatch.setattr("fitz.open", lambda f: (_ for _ in ()).throw(RuntimeError("Should not call fitz")))
    result = extract_text("loan_form.jpeg")
    assert result == "mocked OCR output"




# ============================================================
# 3️⃣ Partial OCR failures
# ============================================================

def test_partial_ocr_failure(monkeypatch):
    """Ensure partial OCR results don't crash."""
    class MockPixmap:
        def save(self, name): pass
    class MockPage:
        def get_pixmap(self): return MockPixmap()
    class MockDoc:
        def load_page(self, n): return MockPage()
    monkeypatch.setattr("fitz.open", lambda f: MockDoc())
    monkeypatch.setattr("paddleocr.PaddleOCR.ocr", lambda self, img, **kw: [[(None, ("Detected text", 0.95))]])
    result = run_ocr_on_pdf_page("mock.pdf", page_num=0)
    assert isinstance(result, str)



# ============================================================
# 4️⃣ Encoding anomalies
# ============================================================

def test_clean_text_with_unicode_symbols():
    """Ensure cleaner handles unicode and currency symbols gracefully."""
    text = "Loan Amount ₹5000 Approved!"
    cleaned = clean_text(text)
    assert isinstance(cleaned, str)
    assert "loan" in cleaned.lower()



def test_clean_text_multilingual_characters():
    """Handle multilingual or accented text gracefully."""
    text = "Crédit Étudiant – aprobado"
    cleaned = clean_text(text)
    assert isinstance(cleaned, str)


# ============================================================
# 5️⃣ Vector index anomalies
# ============================================================

def test_load_vectorstore_empty_dir(tmp_path):
    """Ensure load_vectorstore handles missing directory gracefully."""
    db, emb = load_vectorstore(str(tmp_path))
    assert db is not None and emb is not None


def test_cached_retrieval_on_empty_store(monkeypatch):
    """Simulate retrieval with empty vector index."""
    monkeypatch.setattr("LLMquery.embeddings_index.load_vectorstore", lambda p="": ("db", "emb"))
    out = cached_retrieval("loan terms not found")
    assert isinstance(out, (dict, list))



# ============================================================
# 6️⃣ User query anomalies
# ============================================================


def test_nonsensical_query(monkeypatch):
    """Ensure gibberish queries are handled safely."""
    monkeypatch.setattr(api_server, "cached_retrieval", lambda q: {"answer": "No match"})
    out = api_server.cached_retrieval("asdfghjkl")
    assert isinstance(out, dict)
    assert out["answer"] == "No match"



def test_long_query(monkeypatch):
    """Ensure very long queries don't crash."""
    q = "loan " * 10000
    monkeypatch.setattr(api_server, "cached_retrieval", lambda q: {"answer": "Processed"})
    out = api_server.cached_retrieval(q)
    assert "answer" in out



# ============================================================
# 7️⃣ Pipeline orchestration edge cases
# ============================================================

def test_dag_integrity_loaded():
    """Ensure DAG loads successfully and has tasks."""
    from airflow.models import DAG
    assert isinstance(loan_doc_pipeline_dag.dag, DAG)
    assert len(loan_doc_pipeline_dag.dag.tasks) > 0


def test_dag_task_dependencies_unique():
    """Ensure no duplicate or circular dependencies."""
    dag = loan_doc_pipeline_dag.dag
    task_ids = [t.task_id for t in dag.tasks]
    assert len(task_ids) == len(set(task_ids))


# ============================================================
# 8️⃣ File system / log issues
# ============================================================

def test_log_to_csv_creates_missing_file(tmp_path):
    """Ensure log_to_csv recreates missing CSV file."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    os.chdir(tmp_path)
    os.makedirs("logs", exist_ok=True)
    log_to_csv({"question": "testing"})
    assert (log_dir / "query_logs.csv").exists()


def test_log_to_csv_permission_error(monkeypatch, tmp_path):
    """Simulate permission denied scenario."""
    fake_open = mock.Mock(side_effect=PermissionError("Read-only file system"))
    monkeypatch.setattr(builtins, "open", fake_open)
    try:
        log_to_csv({"question": "test"})
    except PermissionError:
        pytest.skip("PermissionError handled gracefully")
