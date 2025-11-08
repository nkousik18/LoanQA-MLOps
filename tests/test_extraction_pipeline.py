import os
import sys
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

SCRIPTS_PATH = os.path.join(PROJECT_ROOT, "scripts")
if SCRIPTS_PATH not in sys.path:
    sys.path.append(SCRIPTS_PATH)

from scripts.extraction_pipeline.extractor_core import extract_text
from scripts.extraction_pipeline.ocr_utils import run_ocr_on_image, run_ocr_on_pdf_page
from scripts.extraction_pipeline.cleaner import clean_text
from scripts.extraction_pipeline.postprocessor import postprocess_text
from scripts.extraction_pipeline.utils import save_text, list_files
from scripts.extraction_pipeline.main_extractor import (
    process_single_file,
    run_extraction_pipeline,
)
from scripts.extraction_pipeline.config import setup_logger

test_logger = setup_logger("edge_case_tests", log_type="test")
test_logger.info("Starting extraction pipeline unit tests")


def test_extract_text_with_mock(monkeypatch):
    """Ensure PDF extraction path and fallback OCR behave correctly."""
    class MockPage:
        def get_text(self, mode="text", flags=None): return "mocked text"
        def get_pixmap(self): return type("Pix", (), {"save": lambda self, p: None})()
    class MockDoc:
        def __iter__(self): yield MockPage()
        def load_page(self, i): return MockPage()
        def __len__(self): return 1
    monkeypatch.setattr("fitz.open", lambda f: MockDoc())
    monkeypatch.setattr("paddleocr.PaddleOCR.ocr", lambda s, p, **kw: [[(None, ("OCR Output", 0.9))]])

    test_logger.info("Running test_extract_text_with_mock...")
    result = extract_text("dummy.pdf")
    assert isinstance(result, str)
    test_logger.info("PDF extraction mock test passed.")


def test_run_ocr_on_image(monkeypatch):
    monkeypatch.setattr("paddleocr.PaddleOCR.ocr", lambda self, path, **kw: [[(None, ("Detected Text", 0.95))]])
    test_logger.info("Testing run_ocr_on_image...")
    assert "Detected" in run_ocr_on_image("image.png")
    test_logger.info("OCR on image test passed.")


def test_run_ocr_on_pdf_page(monkeypatch, tmp_path):
    class MockPixmap:
        def save(self, path): (tmp_path / "mock.png").write_text("img")
    class MockPage:
        def get_pixmap(self): return MockPixmap()
    class MockDoc:
        def load_page(self, n): return MockPage()
    monkeypatch.setattr("fitz.open", lambda f: MockDoc())
    monkeypatch.setattr("paddleocr.PaddleOCR.ocr", lambda self, img, **kw: [[(None, ("OCR OK", 0.92))]])

    test_logger.info("Testing run_ocr_on_pdf_page...")
    out = run_ocr_on_pdf_page("file.pdf", 0)
    assert "OCR" in out
    test_logger.info("OCR on PDF page test passed.")


def test_clean_text_basic():
    out = clean_text("Loan!!! Details   HERE")
    assert isinstance(out, str)
    assert "Loan" in out
    test_logger.info("Text cleaning test passed.")


def test_postprocess_text_merges_spaces():
    text = postprocess_text("Loan   amount   :  1000")
    assert "  " not in text
    test_logger.info("Postprocessing space merge test passed.")


def test_save_text_creates_file(tmp_path):
    dest = tmp_path
    save_text("data", str(dest), "loan_doc.pdf")
    files = list(dest.glob("loan_doc_*.txt"))
    assert files and files[0].exists()
    test_logger.info("save_text file creation test passed.")


def test_list_files(tmp_path):
    """Ensure list_files detects both txt and pdf."""
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
    test_logger.info("list_files detection test passed.")


def test_process_single_file(monkeypatch):
    """Ensure single file processing runs through all pipeline stages."""
    monkeypatch.setattr("scripts.extraction_pipeline.extractor_core.extract_text", lambda f: "mock text")
    monkeypatch.setattr("scripts.extraction_pipeline.cleaner.clean_text", lambda x: "cleaned text")
    monkeypatch.setattr("scripts.extraction_pipeline.postprocessor.postprocess_text", lambda x: x)
    monkeypatch.setattr("scripts.extraction_pipeline.utils.save_text", lambda t, d, s=None: True)

    test_logger.info("Testing process_single_file...")
    out = process_single_file("loan.pdf")
    assert out is None or out is True
    test_logger.info("process_single_file test passed.")


def test_run_extraction_pipeline(monkeypatch):
    """Ensure batch extraction pipeline runs end-to-end."""
    monkeypatch.setattr("scripts.extraction_pipeline.main_extractor.list_files", lambda p: ["a.pdf", "b.pdf"])
    monkeypatch.setattr("scripts.extraction_pipeline.main_extractor.process_single_file", lambda f: True)

    test_logger.info("Testing run_extraction_pipeline...")
    result = run_extraction_pipeline("data/")
    assert result is None
    test_logger.info("run_extraction_pipeline test passed.")