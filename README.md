# Doc-Understand: End-to-End Loan & Contract Understanding Assistant

## Overview

**Doc-Understand: End-to-End Loan & Contract Understanding Assistant** is designed to help users understand long, technical loan and contract documents by finding important financial terms, understanding legal obligations, and performing translation of content. It takes raw PDFs, performs **Optical Character Recognition (OCR)**, structures the content into line-level spans, builds semantic representation, and provides an interactive assistant. The system is implemented for production-like reliability with cloud storage, multi-stage pipelines, structured logging, and session-based processing.

The overall design includes three major components:

1.  **React Frontend (Vite):** Displays interactive PDFs, text selection, and LLM-generated explanations, translations, and chat responses.
2.  **LLM Backend - vLLM Microservice:** The core intelligence which manages every authenticated **HMAC** request and delivers outputs from language models via endpoints like `/query` and `/chat/query`. It runs the **Mistral-7B Instruct AWQ** model.
3.  **Document Processing Backend (FastAPI):** Handles the PDF upload process, executes **OCR/Textract/DocAI**, and returns structured JSON data that will drive the interactive view of the PDF.

The deployment, titled **LoanDoc Intelligence - Cloud LLM Microservice Deployment**, is executed on **GCP** through the use of a Compute Engine GPU VM.

## Cloud vs. Edge Deployment

The model is deployed on **Google Cloud Platform** with the **Cloud Deployment** strategy. Model inference is served via an instance of a virtual GPU-enabled Compute Engine instance running a **vLLM** server.

## Cloud Deployment (GCP Example)

### Deployment Service

The deployment leverages the following services and technology stack:

* **Compute:** Compute Engine VM (A100/L4 GPU).
* **Inference Server:** **VLLM** (`vllm.entrypoints.openai.api_server`).
* **Microservice:** Flask / FastAPI.
* **Model:** **Mistral 7B Instruct AWQ**.
* **Vector Retrieval:** **RAG** uses **Sentence Transformers** and relies on an in-memory store.
* **Storage:** Uploaded PDFs, OCR data, and embeddings backups are stored on **GCS (Google Cloud Storage)**.
* **Authentication:** Requests are secured via **HMAC-SHA256 signature**.

### Deployment Automation

The entire deployment process is completely automated using **Cloud Build** for **CI/CD**.

### Connection to Repository

The project is version-controlled via **GitHub**. It integrates **Cloud Build** for the management of **CI/CD pipelines**, and the deployment is triggered automatically.

### Detailed Replication Steps

The key steps for the deployment process on the VM are as follows, ensuring that everything is fully reproducible:

1.  **Create GPU VM:** Use the `gcloud compute instances create` command, indicating a zone (e.g., `us-east1-b`), machine type (`g2-standard-8`), and accelerator (`type=nvidia-l4`) with the image family `nvidia-ngc-public`.
2.  **Install Dependencies:** Install essential system packages like `python3-venv`, `git`, and `tmux`.
3.  **VLLM Setup & Deployment:**
    * Install VLLM using `pip install vllm==0.4.2`.
    * Download the **Mistral-7B AWQ** model.
    * Start the vLLM server:
        ```bash
        nohup python3 -m vllm.entrypoints.openai.api_server --port 9000 --quantization=awq --dtype=bfloat16 --gpu-memory-utilization 0.40 &
        ```
4.  **Microservice Deployment:**
    * Install the dependencies for microservices: `flask`, `fastapi`, `uvicorn`, and `sentence-transformers`.
    * Start the microservice:
        ```bash
        nohup uvicorn server:app --host 0.0.0.0 --port 5001 &
        ```
5.  **Test Authentication:** Test that the client computes and sends the required **HMAC signature** properly in the `X-Signature` header, calculated as `HMAC_SHA256(secret, timestamp + "." + body)`.

## Model Monitoring and Triggering Retraining

### Monitoring for Model Decay and Data Shift

Monitoring is implemented through **Cloud Monitoring Dashboards & Alerts** and **Cloud Logging**. Important operational and performance metrics tracked include:

* Response latency
* GPU memory usage
* VLLM token throughput
* Microservice error rate

### Detecting Data Shift

Data drift is detected by observing key features such as **embedding vector norms**, **input token lengths**, and the **domain distribution of the queries**. Custom metrics are created inside **Cloud Monitoring**, to which drift scores are pushed periodically.

### Threshold for Triggering Retraining

The system is designed to trigger retraining automatically when the performance or data drift passes a previously defined threshold:

* **Threshold:** `IF drift_score > 0.45 OR accuracy_drop > 5%`
* **Action Trigger:** The retraining pipeline is triggered using a **Cloud Build** command:
    ```bash
    gcloud builds submit --config retrain.yaml
    ```

### Automating the Retraining Pipeline

This retraining pipeline is fully automated, integrated with the monitoring system, and maintains the performance and accuracy of the model once it is pushed into a production environment. This procedure uses the **GCP services** starting from detecting performance issues to redeploying the updated model without any need for human intervention.

#### 1. Triggering the Automation

The automation is triggered directly by the **Cloud Monitoring Alert**.

* **Source:** **Cloud Monitoring** continuously tracks custom metrics (e.g., `drift_score` and `accuracy_drop`) pushed by the microservice.
* **Condition:** The alert state changes when the threshold is breached (e.g., `drift_score > 0.45 OR accuracy_drop > 5%`).
* **Action:** This change triggers an automated event that runs a **Cloud Build** job.

#### 2. Run the Retraining Pipeline

The alert action is set up to execute an appropriate command in **Cloud Build**, thereby triggering the **CI/CD** pipeline for training:

* **Command:** The system performs a command such as:
    ```bash
    gcloud builds submit --config retrain.yaml
    ```
* **Pipeline Definition:** The `retrain.yaml` contains a complete, automated series of steps needed to create a new version of the model. This includes:
    * **Fetching Data:** Pulls in the newest data along with ground truth labels from **GCS**.
    * **Training:** Running the training scripts for generating new model weights of **Mistral 7B Instruct AWQ**.
    * **Evaluation:** Running a final evaluation against a held-out test set to confirm that the new model meets the minimum accuracy standard.

#### 3. CI/CD for Re-deployment

Once the new model has been trained and its performance has been validated by the job defined in `retrain.yaml`, the already set up **CI/CD chain** is triggered.

* **Artifact Generation:** The updated model weights are saved, and a new container image is built with the updated model.
* **Redeployment:** The new container image is then automatically redeployed by **Cloud Build** to the running **vLLM Microservice** on the **GCP Compute Engine VM** running an A100/L4 GPU. This ensures the improved model version seamlessly switches over and closes the automated monitoring-to-deployment loop.
