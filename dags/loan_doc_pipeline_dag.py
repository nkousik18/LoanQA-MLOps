"""
Airflow DAG for the LoanDoc end-to-end pipeline:
1. OCR extraction
2. Vector index building
3. LLM prompt generation
"""

import os
import glob
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from scripts.extraction_pipeline.config import setup_logger
from scripts.extraction_pipeline.main_extractor import process_single_file
from scripts.LLMquery.build_index import add_to_index
from scripts.LLMquery.prompts.prompt_router import build_prompt, query_ollama
from scripts.LLMquery.prompts.math_utils import evaluate_math


def notify_failure(context):
    """Airflow failure alert callback to log anomalies."""
    task = context.get("task_instance")
    exception = context.get("exception")
    anomaly_logger = setup_logger("airflow_alerts", log_type="anomaly")
    anomaly_logger.error(
        f"Task '{task.task_id}' failed at {context['execution_date']} | DAG: {task.dag_id} | Exception: {exception}"
    )


default_args = {
    "owner": "kousik",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
    "on_failure_callback": notify_failure,
}

dag = DAG(
    dag_id="loan_doc_pipeline_dag",
    description="End-to-end pipeline for loan document extraction, vectorization, and LLM prompt generation",
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["loan", "ocr", "embedding", "llm"],
    default_args=default_args,
)


def extract_task(**_):
    logger = setup_logger("extract_task", log_type="dag")
    anomaly_logger = setup_logger("extract_anomaly", log_type="anomaly")
    logger.info("Starting OCR text extraction...")

    input_dir = "/opt/airflow/data/loan_docs"
    output_dir = "/opt/airflow/data/clean_texts"
    os.makedirs(output_dir, exist_ok=True)

    extracted = []
    for f in os.listdir(input_dir):
        if f.lower().endswith((".pdf", ".png", ".jpg", ".jpeg")):
            path = os.path.join(input_dir, f)
            logger.info(f"Processing file: {f}")
            try:
                out = process_single_file(path)
                if out:
                    extracted.append(out)
                    logger.info(f"Extracted: {out}")
                else:
                    anomaly_logger.warning(f"Empty OCR output for {f}")
            except Exception as e:
                logger.exception(f"Extraction failed for {f}: {e}")
                anomaly_logger.error(f"Extraction anomaly for {f}: {e}")

    logger.info(f"Extraction complete: {len(extracted)} files processed")
    return extracted


extract_op = PythonOperator(
    task_id="extract_text",
    python_callable=extract_task,
    dag=dag,
)


def index_task(**ctx):
    logger = setup_logger("index_task", log_type="dag")
    anomaly_logger = setup_logger("index_anomaly", log_type="anomaly")
    logger.info("Starting vector index update...")

    files = ctx["ti"].xcom_pull(task_ids="extract_text") or []
    updated = 0

    for f in files:
        if os.path.exists(f):
            try:
                add_to_index(f)
                updated += 1
                logger.info(f"Indexed: {f}")
            except Exception as e:
                logger.exception(f"Failed to index {f}: {e}")
                anomaly_logger.error(f"Vectorization anomaly for {f}: {e}")
        else:
            anomaly_logger.warning(f"Missing file reference: {f}")

    logger.info(f"Vector index update complete: {updated} files updated")
    return updated


index_op = PythonOperator(
    task_id="update_vector_index",
    python_callable=index_task,
    dag=dag,
)


def llm_task(**kwargs):
    logger = setup_logger("llm_task", log_type="llm")
    anomaly_logger = setup_logger("llm_anomaly", log_type="anomaly")
    logger.info("Generating LLM prompt via Ollama...")

    from langchain_core.documents import Document

    text_dir = "/opt/airflow/data/clean_texts"
    docs = []

    for path in glob.glob(os.path.join(text_dir, "*.txt")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            docs.append(Document(page_content=text, metadata={"source": os.path.basename(path)}))
            logger.debug(f"Loaded: {os.path.basename(path)} ({len(text)} chars)")
            if len(text) < 40:
                anomaly_logger.warning(f"Low text length anomaly: {os.path.basename(path)}")
        except Exception as e:
            logger.exception(f"Failed to load {path}: {e}")
            anomaly_logger.error(f"Read failure anomaly for {path}: {e}")

    if not docs:
        logger.error("No text files found for LLM processing.")
        anomaly_logger.error("No valid text files available for LLM stage.")
        raise ValueError("No text files found for LLM processing.")

    q = "Can I postpone my federal loan payments?"
    logger.info(f"Building prompt for query: '{q}'")

    try:
        prompt, intent, conf, gap = build_prompt(q, docs, mode=None)
        logger.info(f"Intent: {intent} | Confidence: {conf:.3f} | Gap: {gap:.3f}")

        if conf < 0.6:
            anomaly_logger.warning(f"Low confidence anomaly (Intent={intent}, Conf={conf:.2f})")

        response_text = query_ollama(prompt)

        if not response_text:
            logger.error("Ollama query failed or returned empty response.")
            anomaly_logger.error("Ollama query timeout or no output.")
            raise ValueError("Ollama query failed or timed out.")

        math_check = evaluate_math("10 * 5 + 20")
        logger.info(f"Math sanity check: {math_check}")
        logger.info(f"LLM response (first 250 chars): {response_text[:250]}")

    except Exception as e:
        logger.exception(f"LLM task failed: {e}")
        anomaly_logger.error(f"LLM processing anomaly: {e}")
        raise e


llm_op = PythonOperator(
    task_id="generate_llm_prompt",
    python_callable=llm_task,
    dag=dag,
    retries=2,
    retry_delay=timedelta(minutes=1),
)

extract_op >> index_op >> llm_op

dag