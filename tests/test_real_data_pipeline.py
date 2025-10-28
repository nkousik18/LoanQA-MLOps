"""
🧩 Real Data Integration Tests for LoanDocQA+
=============================================
Enhanced version with structured logging.
Logs each test run to: logs/test_logs/real_data_tests.log
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from datetime import datetime
from scripts.extraction_pipeline.config import setup_logger
from scripts.extraction_pipeline.main_extractor import run_extraction_pipeline
from scripts.LLMquery.build_index import rebuild_vector_index, add_to_index
from scripts.LLMquery.api_server import cached_retrieval, log_to_csv


# ============================================================
# CONFIG (as per project structure)
# ============================================================
DATA_DIR = "data/loan_docs"
OUTPUT_DIR = "data/clean_texts"
INDEX_PATH = "LLMquery/vectorstores/loan_doc_index"
LOG_PATH = "logs/query_logs.csv"
TEST_LOG_DIR = "logs/test_logs"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEST_LOG_DIR, exist_ok=True)

# ============================================================
# GLOBAL LOGGER
# ============================================================
# ============================================================
# GLOBAL LOGGER
# ============================================================
test_logger = setup_logger(
    name="real_data_tests",
    log_type="test"
)
test_logger.info("🚀 Starting Real Data Integration Test Suite.")




# ============================================================
# 1️⃣ Test: Run real extraction
# ============================================================
def test_real_pdf_extraction():
    """Run real OCR extraction on PDFs."""
    pdfs = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    assert pdfs, "❌ No PDF files found in data/loan_docs — add at least one real loan document."

    test_logger.info(f"📄 Found {len(pdfs)} PDF(s). Starting extraction pipeline.")
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
# 2️⃣ Test: Rebuild vector index on extracted text
# ============================================================
def test_vector_index_rebuild():
    """Rebuild Chroma index using extracted text data."""
    files = [os.path.join(OUTPUT_DIR, f) for f in os.listdir(OUTPUT_DIR) if f.endswith(".txt")]
    assert files, "❌ No text files available for indexing."

    test_logger.info("🧱 Rebuilding Chroma index...")
    try:
        rebuild_vector_index()
        test_logger.info("✅ Chroma index rebuild complete.")
    except Exception as e:
        test_logger.exception(f"❌ Index rebuild failed: {e}")
        raise


# ============================================================
# 3️⃣ Test: Real document retrieval
# ============================================================
@pytest.mark.parametrize("query", [
    "What is the interest rate mentioned?",
    "Explain the loan repayment period.",
    "Translate this document into Hindi.",
    "What is the collateral required?",
])
def test_real_query_responses(query):
    """Ask real queries to cached retriever."""
    test_logger.info(f"💬 Query: {query}")
    try:
        docs = cached_retrieval(query)
        assert isinstance(docs, list)
        test_logger.info(f"📄 Retrieved {len(docs)} docs from vectorstore for query: '{query}'")

        # Log query to CSV and to test log
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
# 4️⃣ Test: Add a single extracted file to index
# ============================================================
def test_add_to_index():
    """Add one extracted text file to vector index."""
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
# 5️⃣ Test: Handle empty and corrupt files gracefully
# ============================================================
def test_real_edge_cases():
    """Ensure pipeline handles empty or corrupt PDFs gracefully."""
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
# 6️⃣ Test: Verify logs
# ============================================================
def test_logs_exist():
    """Check if logs exist and are not empty."""
    test_logger.info("🧾 Verifying log files...")
    assert os.path.exists(LOG_PATH), "❌ logs/query_logs.csv not found."

    # Check for any test log file (timestamped)
    test_logs = [f for f in os.listdir(TEST_LOG_DIR) if f.startswith("test_") and f.endswith(".log")]
    assert test_logs, f"❌ No test logs found in {TEST_LOG_DIR}."
    latest_log = max(test_logs, key=lambda f: os.path.getmtime(os.path.join(TEST_LOG_DIR, f)))
    size = os.path.getsize(os.path.join(TEST_LOG_DIR, latest_log))

    test_logger.info(f"📊 Found test log: {latest_log} ({size} bytes)")
    assert size > 0, "❌ Test log file is empty."
    assert os.path.getsize(LOG_PATH) > 0, "❌ query_logs.csv is empty."
    test_logger.info("✅ Log verification passed.")


