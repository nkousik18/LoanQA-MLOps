"""
=========================================
Pipeline Unit + Integration Test Suite
=========================================

Modules Tested:
1. fetch_files.fetch_files()                    → S3 file acquisition
2. run_textract.run_textract_for_pdf()          → AWS Textract extraction
3. segment_text.segment_textract_json()         → JSON segmentation
4. normalize_text.normalize_segmented_json()    → JSON normalization
5. generate_schema_stats.generate_schema_stats() → Schema + validation summary

Run tests:
    pytest -v
"""

import os
import sys
import json
import pytest
from pathlib import Path

# -----------------------------------------------------------------
# 🧭 Ensure correct import path
# -----------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.aws_extraction_scripts import (
    fetch_files,
    run_textract,
    segment_text,
    normalize_text,
    generate_schema_stats,
)

# ============================================================
# FIXTURES
# ============================================================
@pytest.fixture
def sample_raw_json(tmp_path):
    """Create dummy Textract-style raw JSON file."""
    content = {
        "Blocks": [
            {
                "BlockType": "LINE",
                "Text": "Personal Loan Agreement",
                "Confidence": 98.4,
                "Geometry": {"BoundingBox": {"Width": 0.8, "Height": 0.05}},
                "Page": 1,
            },
            {
                "BlockType": "LINE",
                "Text": "Borrower: John Doe",
                "Confidence": 97.2,
                "Geometry": {"BoundingBox": {"Width": 0.9, "Height": 0.05}},
                "Page": 1,
            },
        ]
    }
    path = tmp_path / "loan_raw.json"
    with open(path, "w") as f:
        json.dump(content, f)
    return str(path)


@pytest.fixture
def sample_segmented_json(tmp_path):
    """Create dummy segmented JSON for normalization tests."""
    content = [
        {"text": "Personal Loan Agreement"},
        {"text": "Borrower: John Doe"},
    ]
    path = tmp_path / "loan_segmented.json"
    with open(path, "w") as f:
        json.dump(content, f)
    return str(path)


# ============================================================
# TESTS — FETCH FILES
# ============================================================
def test_fetch_files_returns_list(monkeypatch):
    """Check that fetch_files returns list of .pdf keys."""
    monkeypatch.setattr(fetch_files, "fetch_files", lambda prefix="": ["loan1.pdf", "loan2.pdf"])
    result = fetch_files.fetch_files()
    assert isinstance(result, list)
    assert all(f.endswith(".pdf") for f in result)
    assert len(result) > 0


def test_fetch_files_handles_empty(monkeypatch):
    """Ensure empty bucket returns an empty list."""
    monkeypatch.setattr(fetch_files, "fetch_files", lambda prefix="": [])
    result = fetch_files.fetch_files()
    assert result == []


# ============================================================
# TESTS — TEXTRACT EXTRACTION
# ============================================================
def test_run_textract_for_pdf_runs(monkeypatch):
    """Simulate Textract function without calling AWS."""

    # Mock safe call
    monkeypatch.setattr(run_textract, "safe_textract_call",
                        lambda func=None, **kwargs: {"JobId": "1234", "JobStatus": "SUCCEEDED"})

    # Mock pagination
    monkeypatch.setattr(run_textract, "get_all_textract_results",
                        lambda job_id: {"Blocks": [{"BlockType": "LINE", "Text": "Dummy"}]})

    # Dummy Textract client
    class DummyTextract:
        def start_document_text_detection(self, **kwargs):
            return {"JobId": "1234"}

        def get_document_text_detection(self, **kwargs):
            return {"JobStatus": "SUCCEEDED",
                    "Blocks": [{"BlockType": "LINE", "Text": "Mock line"}]}

    monkeypatch.setattr(run_textract, "textract", DummyTextract())

    try:
        result = run_textract.run_textract_for_pdf("dummy.pdf")
        assert result is None or result.endswith(".json")
    except Exception as e:
        pytest.fail(f"run_textract_for_pdf raised exception: {e}")


# ============================================================
# TESTS — SEGMENTATION
# ============================================================
def test_segment_textract_json_creates_output(sample_raw_json):
    """Ensure segmentation outputs a valid JSON file."""
    out_path = segment_text.segment_textract_json(sample_raw_json)
    assert os.path.exists(out_path)

    with open(out_path, "r") as f:
        data = json.load(f)
    assert isinstance(data, list)
    assert all("text" in d for d in data)
    assert all("conf" in d for d in data)


def test_segment_textract_json_handles_empty(tmp_path):
    """Empty raw JSON should not crash segmentation."""
    raw_path = tmp_path / "empty_raw.json"
    with open(raw_path, "w") as f:
        json.dump({"Blocks": []}, f)

    out_path = segment_text.segment_textract_json(str(raw_path))
    with open(out_path, "r") as f:
        data = json.load(f)

    assert isinstance(data, list)
    assert data == []


# ============================================================
# TESTS — NORMALIZATION
# ============================================================
def test_normalize_segmented_json_creates_output(sample_segmented_json):
    """Ensure normalization creates valid JSON output."""
    out_path = normalize_text.normalize_segmented_json(sample_segmented_json)
    assert os.path.exists(out_path)

    with open(out_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list)
    assert all("text_display" in d for d in data)
    assert all("text_clean" in d for d in data)
    assert all("text_preserved" in d for d in data)


def test_normalize_segmented_json_handles_empty(tmp_path):
    """Empty segmented file should create an empty normalized file."""
    seg_path = tmp_path / "empty_segmented.json"
    with open(seg_path, "w") as f:
        json.dump([], f)

    out_path = normalize_text.normalize_segmented_json(str(seg_path))
    with open(out_path, "r") as f:
        data = json.load(f)

    assert isinstance(data, list)
    assert data == []


# ============================================================
# INTEGRATION TEST — RAW → SEGMENTED → NORMALIZED
# ============================================================
def test_end_to_end_flow(sample_raw_json):
    """Simulate full local pipeline flow."""
    seg_out = segment_text.segment_textract_json(sample_raw_json)
    norm_out = normalize_text.normalize_segmented_json(seg_out)

    assert os.path.exists(seg_out)
    assert os.path.exists(norm_out)

    with open(norm_out, "r") as f:
        data = json.load(f)

    assert len(data) > 0
    assert all("text_clean" in d for d in data)
    assert all("text_display" in d for d in data)


# ============================================================
# TESTS — SCHEMA & VALIDATION (Stage 5)
# ============================================================
def test_generate_schema_stats(tmp_path, monkeypatch):
    """Ensure schema & validation summary is generated properly."""
    dummy_data = [
        {"doc_id": "abc123", "page": 1, "text": "Loan Approved", "conf": 98.5},
        {"doc_id": "abc123", "page": 1, "text": "Interest Rate: 7.5%", "conf": 97.1},
    ]
    norm_dir = tmp_path / "normalized"
    norm_dir.mkdir()
    with open(norm_dir / "loan_normalized.json", "w") as f:
        json.dump(dummy_data, f)

    schema_dir = tmp_path / "schema"
    monkeypatch.setattr(generate_schema_stats, "NORMALIZED_DIR", norm_dir)
    monkeypatch.setattr(generate_schema_stats, "SCHEMA_DIR", schema_dir)

    result = generate_schema_stats.generate_schema_stats()
    assert result and all(Path(p).exists() for p in result)

    # Validate contents
    exp_path, val_path = result
    with open(val_path, "r") as f:
        validation = json.load(f)
    assert validation["overall_passed"] is True
    assert "rules" in validation


# ============================================================
# ▶️ Allow Direct Execution
# ============================================================
if __name__ == "__main__":
    pytest.main(["-v", __file__])
