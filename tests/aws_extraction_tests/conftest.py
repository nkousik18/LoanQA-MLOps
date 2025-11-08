"""
===========================================================
Custom Pytest Config — Offline Dummy Textract Mock
===========================================================

This replaces the real AWS Textract client with a dummy version
so tests run offline. When run directly (▶ button), it executes
all tests and logs the summary result in logs/setup_test.log.
===========================================================
"""

import pytest
import os, sys, datetime

# -----------------------------------------------------------------
# 🧭 Ensure project root is importable (works in VSCode + Airflow)
# -----------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))  # /doc-understand
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.aws_extraction_scripts import run_textract  # ✅ Correct import


# -----------------------------------------------------------------
# 🧪 Dummy Textract Client for Offline Testing
# -----------------------------------------------------------------
class DummyTextract:
    """Mock Textract client for offline testing."""

    def start_document_text_detection(self, **kwargs):
        return {"JobId": "1234"}  # fake job id

    def get_document_text_detection(self, **kwargs):
        return {
            "JobStatus": "SUCCEEDED",
            "Blocks": [{"BlockType": "LINE", "Text": "Mock line from DummyTextract"}],
        }


# -----------------------------------------------------------------
# 🧷 Auto-applied Fixture: Replace Real Textract
# -----------------------------------------------------------------
@pytest.fixture(autouse=True)
def mock_textract(monkeypatch):
    """Replace the real AWS Textract client with DummyTextract."""
    monkeypatch.setattr(run_textract, "textract", DummyTextract())


# -----------------------------------------------------------------
# ▶️ Direct Execution Support
# -----------------------------------------------------------------
if __name__ == "__main__":
    print("\n🚀 Starting Full Test Suite (from conftest.py)\n")

    # Run all tests
    exit_code = pytest.main(["-v", "tests"])

    # Prepare log directory
    log_dir = os.path.join("logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "setup_test.log")

    # Timestamped result
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        if exit_code == 0:
            summary = f"{timestamp} ✅ ALL TESTS PASSED SUCCESSFULLY\n"
            print("\033[92m✅ ALL TESTS PASSED SUCCESSFULLY 🎉\033[0m\n")
        else:
            summary = f"{timestamp} ❌ SOME TESTS FAILED — Check console logs\n"
            print("\033[91m❌ SOME TESTS FAILED — Check logs above.\033[0m\n")
        f.write(summary)

    print(f"🪵 Logged test result → {log_file}\n")
    sys.exit(exit_code)
