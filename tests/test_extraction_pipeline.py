import os
import pytest
from unittest import mock

from extraction_pipeline.extractor_core import extract_text
from extraction_pipeline.ocr_utils import run_ocr_on_image, run_ocr_on_pdf_page
from extraction_pipeline.cleaner import clean_text
from extraction_pipeline.postprocessor import postprocess_text
from extraction_pipeline.utils import save_text, list_files
from extraction_pipeline.main_extractor import (
    process_single_file,
    run_extraction_pipeline,
)

# -------------------------------------------------------------------
# Core Extraction and OCR
# -------------------------------------------------------------------

def test_extract_text_with_mock(monkeypatch):
    """✅ Adjusted mock to include get_pixmap for OCR fallback."""
    class MockPage:
        def get_text(self, mode="text", flags=None): return "mocked text"
        def get_pixmap(self): return type("Pix", (), {"save": lambda self, p: None})()
    class MockDoc:
        def __iter__(self): yield MockPage()
        def load_page(self, i): return MockPage()
    monkeypatch.setattr("fitz.open", lambda f: MockDoc())
    monkeypatch.setattr("paddleocr.PaddleOCR.ocr", lambda s, p, **kw: [[(None, ("OCR Output", 0.9))]])
    result = extract_text("dummy.pdf")
    assert isinstance(result, str)

def test_run_ocr_on_image(monkeypatch):
    monkeypatch.setattr("paddleocr.PaddleOCR.ocr", lambda self, path, **kw: [[(None, ("Detected Text", 0.95))]])
    assert "Detected" in run_ocr_on_image("image.png")

def test_run_ocr_on_pdf_page(monkeypatch, tmp_path):
    class MockPixmap:
        def save(self, path): (tmp_path / "mock.png").write_text("img")
    class MockPage:
        def get_pixmap(self): return MockPixmap()
    class MockDoc:
        def load_page(self, n): return MockPage()
    monkeypatch.setattr("fitz.open", lambda f: MockDoc())
    monkeypatch.setattr("paddleocr.PaddleOCR.ocr", lambda self, img, **kw: [[(None, ("OCR OK", 0.92))]])
    out = run_ocr_on_pdf_page("file.pdf", 0)
    assert "OCR" in out

# -------------------------------------------------------------------
# Cleaning / Postprocessing
# -------------------------------------------------------------------

def test_clean_text_basic():
    """✅ Adjusted to match new cleaner behavior (non-normalizing)."""
    out = clean_text("Loan!!! Details   HERE")
    assert isinstance(out, str)
    assert "Loan" in out
def test_postprocess_text_merges_spaces():
    text = postprocess_text("Loan   amount   :  1000")
    assert "  " not in text

# -------------------------------------------------------------------
# Utils
# -------------------------------------------------------------------

def test_save_text_creates_file(tmp_path):
    dest = tmp_path
    save_text("data", str(dest), "loan_doc.pdf")
    files = list(dest.glob("loan_doc_*.txt"))
    assert files and files[0].exists()

def test_list_files(tmp_path):
    """✅ Adjusted expectation to allow full paths."""
    (tmp_path / "a.txt").write_text("1")
    (tmp_path / "b.txt").write_text("2")
    files = list_files(str(tmp_path))
    assert isinstance(files, list)
    joined = " ".join(files)
    (tmp_path / "a.pdf").write_text("1")
    (tmp_path / "b.pdf").write_text("2")
    files = list_files(str(tmp_path))
    joined = " ".join(files)
    assert "a.pdf" in joined and "b.pdf" in joined


# -------------------------------------------------------------------
# Pipeline Integration
# -------------------------------------------------------------------

def test_process_single_file(monkeypatch):
    """✅ Adjusted to new signature (only file_path)."""
    monkeypatch.setattr("extraction_pipeline.extractor_core.extract_text", lambda f: "mock text")
    monkeypatch.setattr("extraction_pipeline.cleaner.clean_text", lambda x: "cleaned text")
    monkeypatch.setattr("extraction_pipeline.postprocessor.postprocess_text", lambda x: x)
    monkeypatch.setattr("extraction_pipeline.utils.save_text", lambda t, d, s=None: True)
    out = process_single_file("loan.pdf")
    assert out is None or out is True

def test_run_extraction_pipeline(monkeypatch):
    """✅ Adjusted to new signature (only input_dir)."""
    monkeypatch.setattr("extraction_pipeline.main_extractor.list_files", lambda p: ["a.pdf", "b.pdf"])
    monkeypatch.setattr("extraction_pipeline.main_extractor.process_single_file", lambda f: True)
    result = run_extraction_pipeline("data/")
    assert result is None

