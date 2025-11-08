"""
ocr_pipeline_dag.py
-------------------
Airflow DAG for IntelliDoc OCR data pipeline using AWS Textract.

Stages:
1️⃣ Fetch PDFs from S3
2️⃣ Run Textract OCR
3️⃣ Segment JSON
4️⃣ Normalize segmented JSON
5️⃣ Generate schema & validation statistics

All logs and outputs are written to:
  - logs/aws_extraction_logs/
  - data/aws_extraction_data/
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import os, sys

# ---------------------------------------------------------------------
# 🧭 Ensure correct import path (works both locally & inside Docker)
# ---------------------------------------------------------------------
if os.path.exists("/opt/project"):
    # Inside Docker (Airflow container)
    PROJECT_ROOT = "/opt/project"
else:
    # Local run from VS Code
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print(f"📂 Using project root: {PROJECT_ROOT}")

# ---------------------------------------------------------------------
# 📦 Import pipeline stages from your aws_extraction_scripts folder
# ---------------------------------------------------------------------
from scripts.aws_extraction_scripts.fetch_files import fetch_files
from scripts.aws_extraction_scripts.run_textract import run_textract_all
from scripts.aws_extraction_scripts.segment_text import run_segmentation_all
from scripts.aws_extraction_scripts.normalize_text import run_normalization_all
from scripts.aws_extraction_scripts.generate_schema_stats import generate_schema_stats

# ---------------------------------------------------------------------
# ⚙️ Default DAG arguments
# ---------------------------------------------------------------------
default_args = {
    "owner": "intellidoc_team",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# ---------------------------------------------------------------------
# 🧠 Define the Airflow DAG
# ---------------------------------------------------------------------
with DAG(
    dag_id="ocr_pipeline_dag",
    default_args=default_args,
    description="AWS Textract → Segmentation → Normalization → Schema Validation",
    schedule=None,  # manual trigger only
    start_date=datetime(2025, 11, 1),
    catchup=False,
    tags=["ocr", "aws", "textract", "intellidoc"],
) as dag:

    # ---------------------------------------------------------------
    # 1️⃣ Fetch files from S3
    # ---------------------------------------------------------------
    fetch_task = PythonOperator(
        task_id="fetch_files_from_s3",
        python_callable=fetch_files,
    )

    # ---------------------------------------------------------------
    # 2️⃣ Run Textract OCR
    # ---------------------------------------------------------------
    textract_task = PythonOperator(
        task_id="run_textract_ocr",
        python_callable=run_textract_all,
    )

    # ---------------------------------------------------------------
    # 3️⃣ Segment Textract JSON
    # ---------------------------------------------------------------
    segment_task = PythonOperator(
        task_id="segment_textract_json",
        python_callable=run_segmentation_all,
    )

    # ---------------------------------------------------------------
    # 4️⃣ Normalize segmented JSON
    # ---------------------------------------------------------------
    normalize_task = PythonOperator(
        task_id="normalize_segmented_json",
        python_callable=run_normalization_all,
    )

    # ---------------------------------------------------------------
    # 5️⃣ Generate schema & validation statistics
    # ---------------------------------------------------------------
    schema_stats_task = PythonOperator(
        task_id="generate_schema_and_stats",
        python_callable=generate_schema_stats,
    )

    # ---------------------------------------------------------------
    # 🔗 Define pipeline dependencies
    # ---------------------------------------------------------------
    fetch_task >> textract_task >> segment_task >> normalize_task >> schema_stats_task


# ---------------------------------------------------------------------
# 🧩 Debug run (optional)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("🧪 Testing DAG task imports locally...")
    fetch_files()
    print("✅ Local import test successful.")

