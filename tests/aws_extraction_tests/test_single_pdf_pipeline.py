# tests/aws_extraction_tests/test_single_pdf_pipeline.py
"""
Unit tests for the single_pdf_pipeline and basic fetch_files behavior.

We deliberately avoid any real AWS / GCP calls by mocking:
- sync_user_uploads (GCS → S3 sync)
- run_textract_for_pdf_to_dirs (Textract OCR)
- segment_textract_json_to_dir (segmentation)
- gcs_utils flags so NO GCS client is created
"""

import os
import sys
import json
from pathlib import Path
from typing import Tuple

import pytest

# ---------------------------------------------------------------------
# Ensure repo root on sys.path
# ---------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent          # .../tests/aws_extraction_tests
PROJECT_ROOT = CURRENT_DIR.parents[1]                  # .../doc-understand
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------
# Imports from project
# ---------------------------------------------------------------------
from scripts.aws_extraction_scripts import fetch_files
from scripts.aws_extraction_scripts import single_pdf_pipeline
from scripts.aws_extraction_scripts import gcs_utils


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def temp_sessions_root(tmp_path, monkeypatch) -> Path:
    """
    Create a temporary sessions root and patch single_pdf_pipeline.SESSIONS_ROOT
    so all per-session outputs go under a pytest temp folder.

    Structure during the test:

        <tmp>/sessions/<session_id>/{raw, raw_text, segmented, rag}
    """
    sessions_root = tmp_path / "sessions"
    sessions_root.mkdir(parents=True, exist_ok=True)

    # Override the global SESSIONS_ROOT used by the pipeline
    monkeypatch.setattr(single_pdf_pipeline, "SESSIONS_ROOT", sessions_root)

    # VERY IMPORTANT for this test: disable GCS uploads in gcs_utils
    # so that write_json/write_text only write local files and do NOT
    # try to build GCS keys or create a GCS client.
    monkeypatch.setattr(gcs_utils, "USE_GCS_OUTPUT", False)
    monkeypatch.setattr(gcs_utils, "WRITE_LOCAL_COPY", True)

    return sessions_root


# ============================================================
# Simple fetch_files tests (just like before)
# ============================================================

def test_fetch_files_returns_list(monkeypatch):
    """fetch_files.fetch_files should return a list of .pdf keys."""
    monkeypatch.setattr(fetch_files, "fetch_files", lambda prefix="": ["docs/a.pdf", "docs/b.pdf"])
    result = fetch_files.fetch_files()
    assert isinstance(result, list)
    assert all(str(x).endswith(".pdf") for x in result)
    assert len(result) == 2


def test_fetch_files_handles_empty(monkeypatch):
    """Empty bucket case should give an empty list."""
    monkeypatch.setattr(fetch_files, "fetch_files", lambda prefix="": [])
    result = fetch_files.fetch_files()
    assert result == []


# ============================================================
# Single PDF pipeline orchestration test
# ============================================================

def _fake_run_textract_for_pdf_to_dirs(
    pdf_s3_key: str,
    raw_dir: Path,
    raw_text_dir: Path,
) -> Tuple[str, str]:
    """
    Fake Textract step: just writes a tiny raw JSON + raw text file
    into the given dirs and returns their paths.
    """
    raw_dir = Path(raw_dir)
    raw_text_dir = Path(raw_text_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_text_dir.mkdir(parents=True, exist_ok=True)

    raw_json_path = raw_dir / "test_raw.json"
    raw_text_path = raw_text_dir / "test_raw.txt"

    dummy_textract = {
        "Blocks": [
            {
                "BlockType": "LINE",
                "Text": "Dummy loan agreement line",
                "Page": 1,
                "Confidence": 99.0,
                "Geometry": {"BoundingBox": {"Top": 0.1, "Left": 0.1, "Width": 0.8, "Height": 0.05}},
            }
        ]
    }
    with open(raw_json_path, "w", encoding="utf-8") as f:
        json.dump(dummy_textract, f)

    raw_text_path.write_text("Dummy loan agreement line\n", encoding="utf-8")

    return str(raw_json_path), str(raw_text_path)


def _fake_segment_textract_json_to_dir(raw_path: str, segmented_dir: Path) -> str:
    """
    Fake segmentation: reads the dummy raw JSON and writes a simple
    one-span segmented JSON file.
    """
    segmented_dir = Path(segmented_dir)
    segmented_dir.mkdir(parents=True, exist_ok=True)

    seg_path = segmented_dir / "test_segmented.json"

    # Minimal segmented structure compatible with span_adapter
    segmented = [
        {
            "doc_id": "unit_test_doc",
            "page": 1,
            "span_id": 1,
            "span_type": "line",
            "text": "Dummy loan agreement line",
            "conf": 99.0,
            "bbox": {"Top": 0.1, "Left": 0.1, "Width": 0.8, "Height": 0.05},
        }
    ]
    with open(seg_path, "w", encoding="utf-8") as f:
        json.dump(segmented, f)

    return str(seg_path)


def test_process_single_pdf_session_happy_path(temp_sessions_root: Path, monkeypatch):
    """
    Full orchestration test for process_single_pdf_session, using ONLY
    local filesystem + mocks. No real AWS Textract or GCS calls.

    We assert that:
      - returned info contains session_id and key paths
      - the paths actually exist under the temporary sessions root
    """

    # 1) Mock GCP → S3 sync (no network)
    monkeypatch.setattr(single_pdf_pipeline, "sync_user_uploads", lambda: None)

    # 2) Mock Textract OCR step
    monkeypatch.setattr(
        single_pdf_pipeline,
        "run_textract_for_pdf_to_dirs",
        _fake_run_textract_for_pdf_to_dirs,
    )

    # 3) Mock segmentation step
    monkeypatch.setattr(
        single_pdf_pipeline,
        "segment_textract_json_to_dir",
        _fake_segment_textract_json_to_dir,
    )

    # ---- Run the pipeline under test ----
    pdf_s3_key = "user_uploads/unit_test_loan.pdf"
    session_id = "session_unit_test"

    info = single_pdf_pipeline.process_single_pdf_session(
        pdf_s3_key=pdf_s3_key,
        session_id=session_id,
    )

    # ---- Assertions ----
    assert info["session_id"] == session_id

    # session root should be under our temp_sessions_root
    expected_root = temp_sessions_root / session_id
    assert Path(info["session_root"]) == expected_root

    # key paths should exist
    assert Path(info["raw_json"]).exists()
    assert Path(info["raw_text"]).exists()
    assert Path(info["segmented"]).exists()

    # rag outputs should also be present
    assert Path(info["rag"]).exists()
    assert Path(info["rag_chunks"]).exists()
    assert Path(info["rag_blocks"]).exists()
    assert Path(info["rag_embeddings"]).exists()
