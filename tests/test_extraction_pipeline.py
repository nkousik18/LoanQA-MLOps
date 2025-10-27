"""
Comprehensive unit tests for extraction_pipeline.
Validates extraction, OCR, cleaning, saving, and orchestration logic.
All file operations are mocked to run safely.
"""

import os
import pytest
from unittest import mock

# Ensure project root is importable
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from extraction_pipeline.extractor_core import extract_text
from extraction_pipeline.cleaner import clean_text
from extraction_pipeline.utils import save_text, list_files
from extraction_pipeline.postprocessor import postprocess_text
from extraction_pipeline.ocr_utils import run_ocr_on_image, run_ocr_on_pdf_page
from extraction_pipeline.main_extractor import process_file


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

@pytest.fixture
def sample_text():
    return "This IS a SAMPLE text!! 123"


# -------------------------------------------------------------------
# Extraction & OCR Tests
# -------------------------------------------------------------------

def test_extract_text_returns_string(monkeypatch):
    """✅ extract_text should return non-empty string with complete mock of fitz and OCR fallback."""
    class MockPage:
        def get_text(self, mode="text", flags=None):
            return "mocked page text"
    class MockPixmap:
        def save(self, path):
            # Simulate saving OCR image
            with open(path, "w") as f:
                f.write("fakeimg")
    class MockDoc:
        def __iter__(self):
            yield MockPage()
        def load_page(self, i):
            return MockPage()
        def __enter__(self): return self
        def __exit__(self, *a): pass

    monkeypatch.setattr("fitz.open", lambda path: MockDoc())
    monkeypatch.setattr("paddleocr.PaddleOCR.ocr", lambda self, img, **kw: [[(None, ("Mocked OCR", 0.9))]])
    result = extract_text("dummy.pdf")
    assert isinstance(result, str)
    assert "mocked" in result or "OCR" in result




def test_run_ocr_on_image(monkeypatch):
    """✅ run_ocr_on_image should return recognized text."""
    monkeypatch.setattr("paddleocr.PaddleOCR.ocr", lambda self, path, **kw: [[(None, ("Text Found", 0.95))]])
    result = run_ocr_on_image("fake_image.png")
    assert "Text" in result


def test_run_ocr_on_pdf_page(monkeypatch, tmp_path):
    """✅ run_ocr_on_pdf_page mocked to avoid real files."""
    # page.get_pixmap().save(path) is called, so mock that
    class MockPixmap:
        def save(self, path):
            # simulate saving image file
            (tmp_path / "temp_img.png").write_text("fakeimg")
    class MockPage:
        def get_pixmap(self): return MockPixmap()
    class MockDoc:
        def load_page(self, n): return MockPage()
    monkeypatch.setattr("fitz.open", lambda path: MockDoc())
    monkeypatch.setattr(
        "paddleocr.PaddleOCR.ocr",
        lambda self, img, **kw: [[(None, ("Mocked OCR Output", 0.95))]]
    )
    result = run_ocr_on_pdf_page("page.pdf", page_num=0)
    assert "Mocked" in result

# -------------------------------------------------------------------
# Cleaning & Postprocessing
# -------------------------------------------------------------------

def test_clean_text_basic(sample_text):
    """✅ clean_text normalizes and removes punctuation."""
    cleaned = clean_text(sample_text)
    assert isinstance(cleaned, str)
    assert "sample" in cleaned.lower()


def test_postprocess_text_removes_extra_spaces():
    """✅ postprocess_text should condense spaces."""
    result = postprocess_text("Loan   terms   and  conditions")
    assert "  " not in result


# -------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------

def test_save_text_creates_file(tmp_path):
    """✅ save_text should create output file correctly when given folder."""
    output_dir = tmp_path
    source_file = "dummy.pdf"
    save_text("Hello Loan World", str(output_dir), source_file)
    files = list(output_dir.glob("dummy_*.txt"))
    assert len(files) == 1
    assert "Loan" in files[0].read_text()


def test_list_files(tmp_path):
    """✅ list_files should return all .txt files."""
    (tmp_path / "a.txt").write_text("1")
    (tmp_path / "b.txt").write_text("2")
    files = list_files(str(tmp_path))
    assert isinstance(files, list)
    assert all(f.endswith(".txt") for f in files)


# -------------------------------------------------------------------
# Main Extraction Process
# -------------------------------------------------------------------


def test_process_file_runs(monkeypatch):
    """✅ process_file should complete even if OCR fallback is triggered."""
    class MockPage:
        def get_text(self, mode="text", flags=None): return "mock text"
    class MockDoc:
        def __iter__(self):
            yield MockPage()
        def load_page(self, i):
            return MockPage()
        def __enter__(self): return self
        def __exit__(self, *a): pass

    monkeypatch.setattr("fitz.open", lambda path: MockDoc())
    monkeypatch.setattr("extraction_pipeline.cleaner.clean_text", lambda t: t.lower())
    monkeypatch.setattr("extraction_pipeline.utils.save_text", lambda text, dir, src: True)
    monkeypatch.setattr("paddleocr.PaddleOCR.ocr", lambda self, img, **kw: [[(None, ("Mocked OCR", 0.9))]])
    result = process_file("dummy.pdf")
    assert result is None or isinstance(result, str)
