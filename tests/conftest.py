"""
🌍 Final Unified Test Setup for LoanDocQA+
==========================================
Ensures all tests run with:
 - valid DAG file
 - real sample PDF
 - clean_texts directory + dummy text
 - seeded test logs
"""

import os, io, pytest
from datetime import datetime
from reportlab.pdfgen import canvas  # ✅ Generates a real PDF for PyMuPDF

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)

# ============================================================
# Ensure Required Directories Exist
# ============================================================
REQUIRED = ["dags", "data/loan_docs", "data/clean_texts", "logs/test_logs"]
for d in REQUIRED:
    os.makedirs(os.path.join(ROOT, d), exist_ok=True)

# ============================================================
# 1️⃣ Create Minimal DAG File
# ============================================================
dag_path = os.path.join(ROOT, "dags/loan_doc_pipeline_dag.py")
if not os.path.exists(dag_path):
    with open(dag_path, "w", encoding="utf-8") as f:
        f.write(
            "from airflow import DAG\n"
            "from datetime import datetime\n"
            "from airflow.operators.empty import EmptyOperator\n\n"
            "with DAG('loan_doc_pipeline_dag', start_date=datetime(2024,1,1), schedule=None, catchup=False):\n"
            "    start = EmptyOperator(task_id='start')\n"
            "    end = EmptyOperator(task_id='end')\n"
            "    start >> end\n"
        )

# ============================================================
# 2️⃣ Create a Valid, Readable Sample PDF
# ============================================================
pdf_path = os.path.join(ROOT, "data/loan_docs/sample.pdf")
if not os.path.exists(pdf_path):
    from reportlab.lib.pagesizes import letter
    c = canvas.Canvas(pdf_path, pagesize=letter)
    c.drawString(100, 700, "Loan Agreement Document")
    c.drawString(100, 680, "Interest Rate: 7.5%")
    c.drawString(100, 660, "Repayment period: 24 months")
    c.save()

# ============================================================
# 3️⃣ Create Dummy Extracted Text File
# ============================================================
txt_path = os.path.join(ROOT, "data/clean_texts/sample_extracted.txt")
if not os.path.exists(txt_path):
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Sample loan text extracted successfully.\n")

# ============================================================
# 4️⃣ Ensure Logs Exist
# ============================================================
query_log = os.path.join(ROOT, "logs/query_logs.csv")
if not os.path.exists(query_log):
    with open(query_log, "w", encoding="utf-8") as f:
        f.write("timestamp,question,intent,confidence,gap,mode,response_length,time_taken_sec,sources,prompt,answer\n")

seed_log = os.path.join(ROOT, "logs/test_logs/seed_test.log")
if not os.path.exists(seed_log):
    with open(seed_log, "w", encoding="utf-8") as f:
        f.write(f"[{datetime.now()}] Test environment initialized.\n")

# ============================================================
# Diagnostics
# ============================================================
def pytest_sessionstart(session):
    print(f"\n[pytest] DAG ready: {os.path.exists(dag_path)} | PDF valid: {os.path.exists(pdf_path)} | CleanText: {os.path.exists(txt_path)}")
    print(f"[pytest] Logs ready: {os.path.exists(seed_log)}")

@pytest.fixture(scope="session", autouse=True)
def ensure_env():
    os.chdir(ROOT)
    yield
    os.chdir(ROOT)
