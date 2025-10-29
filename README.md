
# LoanQA-MLOps: End-to-End Loan Document Intelligence Pipeline

##  Overview

**LoanQA-MLOps** is a  machine learning data pipeline built to automate the extraction, processing, and intelligent querying of financial and loan related documents using OCR, vector embeddings, and LLM-based retrieval.

The project demonstrates a full MLOps workflow from **data acquisition to orchestration, testing, logging, anomaly detection, and reproducibility** implemented through **Airflow DAGs**, **Docker**, and **DVC**.

### Detailed User Needs Worksheet:

[https://docs.google.com/document/d/12dYbbpB0W6WBYzO4yDMSRsQaNEjjQXcYo61cStaCd0k/edit?usp=sharing]

### Detailed Errors and Failure :
[https://docs.google.com/document/d/12dYbbpB0W6WBYzO4yDMSRsQaNEjjQXcYo61cStaCd0k/edit?usp=sharing]

### Data Pipeline for API approach:
[https://docs.google.com/document/d/1iw2rjK1tuYKPVX7fq9yl6H48UkradVXj9BYgm_qvBGY/edit?usp=sharing]

### Data pipeline for Local implementation:
[https://docs.google.com/document/d/1BWzV0JR8U0b1pw3ZfCMO6ESoFkCXYUV22zRPGZlIXcY/edit?usp=sharing]


---

##  Environment Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/nkousik18/LoanQA-MLOps.git
cd LoanQA-MLOps
```

### 2️⃣ Create and Activate Virtual Environment

Using **conda**:

```bash
conda create -n loanqa_env python=3.10 -y
conda activate loanqa_env
```

Or using **venv**:

```bash
python3 -m venv venv
source venv/bin/activate      # (Mac/Linux)
venv\Scripts\activate         # (Windows)
```

### 3️⃣ Install Dependencies

All dependencies are listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

If using Airflow via Docker:

```bash
docker-compose up -d --build
```

### 4️⃣ Configure Environment Variables

Create a `.env` file in the root directory:

```
DATA_PATH=data/loan_docs
CLEAN_TEXT_PATH=data/clean_texts
LOG_DIR=logs
VECTOR_DB_PATH=scripts/LLMquery/vectorstores/local_doc_index
MODEL_NAME=all-MiniLM-L6-v2
```

### 5️⃣  Initialize DVC for Data Versioning

```bash
dvc init
dvc remote add -d local_storage ./data
dvc add data/loan_docs
git add data/loan_docs.dvc .gitignore
git commit -m "Data versioning initialized"
dvc push
```

---

##  Running the Pipeline

### **Option A: Run Through Airflow Orchestration**

1. Launch Airflow services:

   ```bash
   docker-compose up -d
   ```
2. Access Airflow UI at `http://localhost:8080`
3. Trigger the DAG: **loan_doc_pipeline_dag**

**DAG Flow:**

```
[OCR Extraction] → [Vector Index Build] → [LLM Prompt Generation]
```

### **Option B: Run Modules Individually (Local Testing)**

```bash
# Step 1: OCR Extraction
python scripts/extraction_pipeline/main_extractor.py --input data/loan_docs --output data/clean_texts

# Step 2: Vector Index Build
python scripts/LLMquery/build_index.py --input data/clean_texts --index scripts/LLMquery/vectorstores/local_doc_index

# Step 3: LLM Query / Prompt Testing
python scripts/LLMquery/prompts/prompt_router.py
```

### **Option C: Run All Tests**

```bash
pytest tests/ -v --maxfail=1 --disable-warnings
```

---

## 🧩 Code Structure

```plaintext
LoanQA-MLOps/
│
├── dags/                                  # Airflow orchestration DAGs
│   └── loan_doc_pipeline_dag.py           # Main Loan Document pipeline DAG
│
├── data/                                  # Data storage and sources
│   ├── loan_docs/                         
│   │   ├── 1. Introduction.pdf
│   │   ├── user-needs.pdf
│   │   ├── corrupt.pdf
│   │   └── empty.pdf
│   ├── handwritten_forms/                 # Sample handwritten loan forms
│   ├── clean_texts/                       # OCR-extracted and cleaned text files
│   ├── financial_terms.csv                # Finance keyword reference data
│   └── Finance_terms_definitions_labels.csv
│
├── scripts/                               # Core pipeline scripts
│   ├── extraction_pipeline/               # OCR extraction and preprocessing
│   │   ├── main_extractor.py              # Entry point for extraction
│   │   ├── extractor_core.py              # Handles OCR + parsing
│   │   ├── cleaner.py                     # Cleans extracted text
│   │   ├── postprocessor.py               # Normalizes and structures data
│   │   ├── config.py                      # Logging and configuration setup
│   │   ├── ocr_utils.py, utils.py         # Utility functions
│   │
│   ├── LLMquery/                          # Vector index + LLM querying modules
│   │   ├── build_index.py                 # Builds embedding vector index
│   │   ├── api_server.py                  # Backend API for LLM queries
│   │   ├── embeddings_index.py            # Manages embedding storage
│   │   ├── inspect_index.py               # Index diagnostics
│   │   ├── prompts/                       # LLM prompt templates & math utils
│   │   ├── static/                        # Frontend/static assets (if used)
│   │   └── vectorstores/                  # Local ChromaDB index store
│   │
│   └── setup.py                           # Local package installer for scripts
│
├── tests/                                 # Automated test suite
│   ├── test_dags.py                       # DAG structure and dependency tests
│   ├── test_extraction_pipeline.py        # OCR and preprocessing validation
│   ├── test_LLMquery_pipeline.py          # Vectorization and LLM QA tests
│   ├── test_performance_metrics.py        # Throughput and performance checks
│   ├── test_real_data_pipeline.py         # End-to-end pipeline test
│   ├── test_edge_cases.py                 # Corrupted/missing data tests
│   └── conftest.py                        # Pytest configuration
│
├── logs/                                  # Centralized logging and monitoring
│   ├── extraction_logs/                   # OCR and preprocessing logs
│   ├── llm_logs/                          # LLM query execution logs
│   ├── test_logs/                         # Pytest and performance logs
│   ├── anomaly_logs/                      # Detected anomalies and alerts
│   ├── scheduler/                         # Airflow scheduler output
│   └── dag_logs/, dag_processor_manager/  # Airflow process logs
│
├── docker-compose.yaml                    # Airflow orchestration setup
├── Dockerfile.airflow                     # Airflow image configuration
├── requirements.txt                       # Python dependencies
├── README.md                              # Project documentation
└── dvc.yaml (optional)                    # Data versioning (if using DVC)

```

---

## Tracking & Logging

* **Custom Loggers** implemented in `scripts/extraction_pipeline/config.py` track every stage (timestamped).
* Logs stored under `logs/` for **preprocessing, LLM queries, DAG scheduler, and tests**.
* Airflow also captures task-level logs viewable from the UI.

---

##  Anomaly Detection & Alerts

* Monitors OCR extraction failures, corrupted PDFs, and vector build anomalies.
* Automatically logs discrepancies in `logs/performance_*.log` and triggers alerts.

---

##  Reproducibility Details

1. **Environment Reproducibility:**
   All dependencies are listed in `requirements.txt` and pinned to versions for deterministic builds.

2. **Data Versioning:**
   Handled using **DVC** (`.dvc` files track changes in `data/loan_docs`).

3. **Logging Reproducibility:**
   Each run creates timestamped logs under `logs/extraction_logs/` and `logs/llm_logs/`  ensuring traceable results.

4. **Pipeline Reproducibility:**
   Airflow DAGs ensure the same sequence of tasks is executed across environments.

5. **Testing Reproducibility:**
   Pytest modules validate consistent outcomes for OCR accuracy, index integrity, and LLM response stability.

---

## Testing and Validation

Run all automated tests to validate pipeline integrity:

```bash
pytest tests/ --cov=scripts --cov-report=term-missing
```

Key validations include:

* DAG structure and dependency correctness
* OCR extraction accuracy
* Vector store performance and latency
* End-to-end pipeline throughput

---

## Deployment Notes

* Built with **modular components** to allow easy extension (e.g., new document types or LLM models).
* Airflow orchestration ensures scalable and fault-tolerant execution.
* Future versions will integrate **Fine-Tuning** and **RAG optimization** using the accumulated document corpus.

---

##  Authors

**Kousik Nandury**
Graduate Student, Northeastern University
Email: [nandury.k@northeastern.edu]

**Yaswanth Kumar Reddy Gujjula**
Graduate Student, Northeastern University
Email: [gujjula.y@northeastern.edu]



---
