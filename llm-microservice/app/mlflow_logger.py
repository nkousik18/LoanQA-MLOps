import mlflow
import time
import json
from app.config import settings


def init_mlflow():
    """
    Initializes MLflow configuration.
    """
    mlflow.set_tracking_uri("http://localhost:5001")  # Your MLflow server URI
    mlflow.set_experiment("LoanDocAI-LLM-Inference")


def log_llm_query(payload, result, timing, retrieved_chunks):
    """
    Logs a single query into MLflow.
    """
    init_mlflow()

    with mlflow.start_run(run_name="llm_query"):
        # -----------------------------
        # INPUTS
        # -----------------------------
        mlflow.log_text(json.dumps(payload, indent=2), "input/payload.json")

        # Router decisions
        mlflow.log_param("router_mode", result.get("mode"))
        mlflow.log_metric("router_confidence", result.get("router_confidence", 0))

        # -----------------------------
        # RETRIEVAL INFO
        # -----------------------------
        mlflow.log_metric("retrieval_confidence", result.get("retrieval_confidence", 0))
        mlflow.log_metric("retrieved_chunks_count", len(retrieved_chunks))

        mlflow.log_text(
            json.dumps(retrieved_chunks, indent=2),
            "retrieval/retrieved_chunks.json"
        )

        # -----------------------------
        # MODEL OUTPUT
        # -----------------------------
        mlflow.log_text(result.get("response", ""), "output/response.txt")

        # -----------------------------
        # PERFORMANCE
        # -----------------------------
        mlflow.log_metric("retrieval_ms", timing.get("retrieval_ms", 0))
        mlflow.log_metric("router_ms", timing.get("router_ms", 0))
        mlflow.log_metric("llm_ms", timing.get("llm_ms", 0))
        mlflow.log_metric("total_ms", timing.get("total_ms", 0))

        # Timestamp for easy querying
        mlflow.log_param("timestamp", int(time.time()))


def log_error(error_message, payload):
    init_mlflow()

    with mlflow.start_run(run_name="llm_error"):
        mlflow.log_param("error", str(error_message))
        mlflow.log_text(json.dumps(payload, indent=2), "input/payload.json")
