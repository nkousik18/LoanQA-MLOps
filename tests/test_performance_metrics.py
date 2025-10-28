"""
📊 Performance Benchmark Suite — LoanDocQA+ Pipeline
====================================================
This script measures:
    - OCR extraction latency per PDF
    - Vector index rebuild latency
    - Retrieval (query) latency
    - Overall pipeline throughput (files/minute)

Structured logs → logs/performance_logs/
Metrics CSV → logs/performance_metrics.csv
"""

import os
import time
import csv
import pytest
from datetime import datetime
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.extraction_pipeline.config import setup_logger
from scripts.extraction_pipeline.main_extractor import run_extraction_pipeline
from scripts.LLMquery.build_index import rebuild_vector_index
from scripts.LLMquery.api_server import cached_retrieval, log_to_csv


# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = "data/loan_docs"
OUTPUT_DIR = "data/clean_texts"
INDEX_PATH = "LLMquery/vectorstores/loan_doc_index"
LOGS_DIR = "logs/test_logs"
PERF_LOG = os.path.join(LOGS_DIR, "performance_metrics.csv")
PERF_LOG_DIR = os.path.join(LOGS_DIR, "performance_logs")

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(PERF_LOG_DIR, exist_ok=True)

# Initialize structured logger
perf_logger = setup_logger(name="performance_tests", log_type="performance")
perf_logger.info("🚀 Starting Performance Benchmark Suite")

# CSV header
HEADER = [
    "timestamp", "metric", "description",
    "num_files", "duration_sec", "throughput_files_per_min"
]


def _log_metric(metric, description, num_files, duration):
    """Append benchmark result to CSV and structured log."""
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "metric": metric,
        "description": description,
        "num_files": num_files,
        "duration_sec": round(duration, 2),
        "throughput_files_per_min": round((num_files / duration * 60), 2) if duration > 0 else 0,
    }

    file_exists = os.path.exists(PERF_LOG)
    with open(PERF_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    perf_logger.info(
        f"✅ {metric}: {duration:.2f}s | "
        f"{row['throughput_files_per_min']} files/min | "
        f"Files: {num_files} | Desc: {description}"
    )


# ============================================================
# 1️⃣ OCR + Extraction Time
# ============================================================
def test_ocr_extraction_performance():
    """Measure how long OCR extraction takes for all PDFs."""
    pdfs = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    assert pdfs, "❌ No PDFs in data/loan_docs to benchmark."

    perf_logger.info(f"🧾 Starting OCR extraction for {len(pdfs)} PDF(s)...")
    start = time.perf_counter()
    run_extraction_pipeline(DATA_DIR)
    end = time.perf_counter()
    duration = end - start

    _log_metric("OCR_Extraction", f"OCR extraction for {len(pdfs)} PDFs", len(pdfs), duration)
    perf_logger.info(f"✅ OCR extraction completed in {duration:.2f}s.")


# ============================================================
# 2️⃣ Vector Index Build Time
# ============================================================
def test_vector_index_build_performance():
    """Measure time to rebuild Chroma index."""
    txts = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".txt")]
    assert txts, "❌ No extracted text files for embedding."

    perf_logger.info(f"🔗 Rebuilding vector index from {len(txts)} text files...")
    start = time.perf_counter()
    rebuild_vector_index()
    end = time.perf_counter()

    duration = end - start
    _log_metric("Vector_Index_Rebuild", "Embedding + persistence of extracted docs", len(txts), duration)
    perf_logger.info(f"✅ Vector index rebuild took {duration:.2f}s.")


# ============================================================
# 3️⃣ Retrieval Latency (avg of multiple queries)
# ============================================================
@pytest.mark.parametrize("query", [
    "What is the interest rate mentioned?",
    "Explain the loan repayment period.",
    "What is the collateral required?",
])
def test_retrieval_latency(query):
    """Measure retrieval latency per query."""
    perf_logger.info(f"💬 Measuring retrieval latency for: '{query}'")
    start = time.perf_counter()
    docs = cached_retrieval(query)
    end = time.perf_counter()
    latency = end - start

    assert isinstance(docs, list)
    _log_metric("Retrieval_Latency", f"Query='{query[:30]}...'", len(docs), latency)
    perf_logger.info(f"✅ Retrieval latency for '{query}' = {latency:.2f}s.")


# ============================================================
# 4️⃣ Overall Throughput (Extraction + Index)
# ============================================================
def test_total_pipeline_throughput():
    """Aggregate overall throughput from extraction + index steps."""
    if not os.path.exists(PERF_LOG):
        pytest.skip("No performance data recorded yet.")

    total_files = 0
    total_time = 0.0
    with open(PERF_LOG, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["metric"] in ["OCR_Extraction", "Vector_Index_Rebuild"]:
                total_files += int(row["num_files"])
                total_time += float(row["duration_sec"])

    assert total_time > 0, "❌ No recorded timings found."
    throughput = round((total_files / total_time * 60), 2)

    _log_metric("Pipeline_Throughput", "End-to-end extraction + indexing", total_files, total_time)
    perf_logger.info(f"🚀 Overall Throughput: {throughput} files/minute")
