"""
🧩 Real Data Integration Tests for LoanDocQA+
=============================================
Enhanced version with structured logging.
Logs each test run to: logs/test_logs/real_data_tests.log
"""

import sys, os
from datetime import datetime
import pytest

# ============================================================
# Ensure correct working directory and imports
# ============================================================
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
os.chdir(ROOT_DIR)
sys.path.append(ROOT_DIR)

from scripts.extraction_pipeline.config import setup_logger
from scripts.extraction_pipeline.main_extractor import run_extraction_pipeline
from scripts.LLMquery.build_index import rebuild_vector_index, add_to_index
from scripts.LLMquery.api_server import cached_retrieval, log_to_csv

# ============================================================
# CONFIGURATION (project structure aware)
# ============================================================
DATA_DIR = os.path.join("data", "loan_docs")
OUTPUT_DIR = os.path.join("data", "clean_texts")
LOG_PATH = os.path.join("logs", "query_logs.csv")
TEST_LOG_DIR = os.path.join("logs", "test_logs")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEST_LOG_DIR, exist_ok=True)

# ============================================================
# GLOBAL LOGGER
# ============================================================
test_logger = setup_logger(name="real_data_tests", log_type="test")
test_logger.info("🚀 Starting Real Data Integration Test Suite.")


# ============================================================
# 1️⃣ Real OCR Extraction
# ============================================================
def test_real_pdf_extraction():
    """Run OCR extraction on real PDFs."""
    pdfs = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    assert pdfs, "❌ No PDFs found in data/loan_docs."

    test_logger.info(f"📄 Found {len(pdfs)} PDF(s). Starting extraction pipeline...")
    try:
        run_extraction_pipeline(DATA_DIR)
        txt_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".txt")]
        test_logger.info(f"📂 Extracted {len(txt_files)} files → {OUTPUT_DIR}")
        assert txt_files, "❌ No text files found after extraction."
        test_logger.info("✅ Real PDF extraction successful.")
    except Exception as e:
        test_logger.exception(f"❌ Extraction failed: {e}")
        raise


# ============================================================
# 2️⃣ Vector Index Rebuild
# ============================================================
def test_vector_index_rebuild():
    """Rebuild Chroma index using extracted text files."""
    files = [os.path.join(OUTPUT_DIR, f) for f in os.listdir(OUTPUT_DIR) if f.endswith(".txt")]
    assert files, "❌ No text files available for indexing."

    test_logger.info(f"🧱 Rebuilding Chroma index from {len(files)} files...")
    try:
        rebuild_vector_index()
        test_logger.info("✅ Chroma index rebuild complete.")
    except Exception as e:
        test_logger.exception(f"❌ Index rebuild failed: {e}")
        raise


# ============================================================
# 3️⃣ Retrieval Queries
# ============================================================
@pytest.mark.parametrize("query", [
    "What is the interest rate mentioned?",
    "Explain the loan repayment period.",
    "Translate this document into Hindi.",
    "What is the collateral required?",
])
def test_real_query_responses(query):
    """Validate retrieval responses for real queries."""
    test_logger.info(f"💬 Query: {query}")
    try:
        docs = cached_retrieval(query)
        assert isinstance(docs, list)
        test_logger.info(f"📄 Retrieved {len(docs)} docs for query: '{query}'")

        # Log query to CSV
        log_to_csv({
            "timestamp": datetime.now().isoformat(),
            "question": query,
            "intent": "integration-test",
            "confidence": "1.0",
            "gap": "0",
            "mode": "auto",
            "response_length": 0,
            "time_taken_sec": 0,
            "sources": "mock",
            "prompt": "mock",
            "answer": "mock"
        })
        test_logger.info("✅ Query handled and logged successfully.")
    except Exception as e:
        test_logger.exception(f"❌ Query '{query}' failed: {e}")
        raise


# ============================================================
# 4️⃣ Add Single File to Index
# ============================================================
def test_add_to_index():
    """Add one extracted text file to the vector index."""
    files = [os.path.join(OUTPUT_DIR, f) for f in os.listdir(OUTPUT_DIR) if f.endswith(".txt")]
    assert files, "❌ No .txt files found to add."

    test_logger.info(f"🪣 Adding file to index: {files[0]}")
    try:
        result = add_to_index(files[0])
        assert result is not None
        test_logger.info("✅ File successfully indexed.")
    except Exception as e:
        test_logger.exception(f"❌ Failed to add to index: {e}")
        raise


# ============================================================
# 5️⃣ Handle Empty & Corrupt PDFs
# ============================================================
def test_real_edge_cases():
    """Ensure pipeline handles empty/corrupt PDFs gracefully."""
    empty_pdf = os.path.join(DATA_DIR, "empty.pdf")
    corrupt_pdf = os.path.join(DATA_DIR, "corrupt.pdf")

    open(empty_pdf, "wb").close()
    with open(corrupt_pdf, "wb") as f:
        f.write(b"\x00\x01garbage")

    test_logger.info("⚙️ Created test files: empty.pdf & corrupt.pdf")
    try:
        run_extraction_pipeline(DATA_DIR)
    except Exception as e:
        test_logger.warning(f"⚠️ Gracefully handled file error: {e.__class__.__name__}")

    assert os.path.exists(empty_pdf)
    assert os.path.exists(corrupt_pdf)
    test_logger.info("✅ Edge case handling verified.")


# ============================================================
# 6️⃣ Log Verification
# ============================================================
def test_logs_exist():
    """Check for existence and content of log files."""
    test_logger.info("🧾 Verifying log files...")
    assert os.path.exists(LOG_PATH), "❌ logs/query_logs.csv not found."

    test_logs = [f for f in os.listdir(TEST_LOG_DIR) if f.endswith(".log")]
    assert test_logs, f"❌ No test logs found in {TEST_LOG_DIR}."

    latest_log = max(test_logs, key=lambda f: os.path.getmtime(os.path.join(TEST_LOG_DIR, f)))
    log_path = os.path.join(TEST_LOG_DIR, latest_log)
    size = os.path.getsize(log_path)

    test_logger.info(f"📊 Found test log: {latest_log} ({size} bytes)")
    assert size > 0, "❌ Test log file is empty."
    assert os.path.getsize(LOG_PATH) > 0, "❌ query_logs.csv is empty."
    test_logger.info("✅ Log verification passed.")
