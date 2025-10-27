from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import os
import sys
import glob

# Minimal imports here so Airflow's parser can load quickly
sys.path += ["/opt/airflow/extraction_pipeline", "/opt/airflow/LLMquery"]

default_args = {
    "owner": "kousik",
    "depends_on_past": False,
    "email_on_failure": True,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="loan_doc_pipeline_dag",
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["loan", "ocr", "embedding"],
    default_args=default_args,
) as dag:

    # ---------------------- #
    # 1️⃣  Extract Text Task  #
    # ---------------------- #
    def extract_task(**_):
        from extraction_pipeline.main_extractor import process_single_file

        input_dir = "/opt/airflow/data/loan_docs"
        output_dir = "/opt/airflow/data/clean_texts"
        os.makedirs(output_dir, exist_ok=True)

        extracted = []
        for f in os.listdir(input_dir):
            if f.lower().endswith((".pdf", ".png", ".jpg", ".jpeg")):
                path = os.path.join(input_dir, f)
                out = process_single_file(path)
                if out:
                    extracted.append(out)
        print(f"✅ Extracted {len(extracted)} documents")
        return extracted

    extract_op = PythonOperator(
        task_id="extract_text",
        python_callable=extract_task,
    )

    # ---------------------- #
    # 2️⃣  Build Vector Index #
    # ---------------------- #
    def index_task(**ctx):
        from LLMquery.build_index import add_to_index

        files = ctx["ti"].xcom_pull(task_ids="extract_text") or []
        updated = 0
        for f in files:
            if os.path.exists(f):
                add_to_index(f)
                updated += 1
        print(f"✅ Vector index updated for {updated} files")

    index_op = PythonOperator(
        task_id="update_vector_index",
        python_callable=index_task,
    )

    # ---------------------- #
    # 3️⃣  Generate LLM Prompt #
    # ---------------------- #
    def llm_task():
        from LLMquery.prompts.prompt_router import build_prompt
        from LLMquery.prompts.math_utils import evaluate_math
        from langchain_core.documents import Document

        text_dir = "/opt/airflow/data/clean_texts"
        docs = []

        # Load all extracted text files as LangChain Documents
        for path in glob.glob(os.path.join(text_dir, "*.txt")):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            docs.append(Document(page_content=text, metadata={"source": os.path.basename(path)}))

        if not docs:
            raise ValueError("No text files found for LLM processing.")

        q = "Can I postpone my federal loan payments?"
        prompt, intent, conf, gap = build_prompt(q, docs, mode=None)

        print("✅ Prompt built successfully.")
        print(f"Intent → {intent}")
        print(f"Confidence → {conf:.3f}")
        print(f"Math sanity check → {evaluate_math('10 * 5 + 20')}")

    llm_op = PythonOperator(
        task_id="generate_llm_prompt",
        python_callable=llm_task,
    )

    # Orchestrate dependencies
    extract_op >> index_op >> llm_op
