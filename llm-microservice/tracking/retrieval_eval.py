# tracking/retrieval_eval.py
import mlflow
from typing import List, Dict, Any


def run_retrieval_experiment(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    retrieval_time_ms: float,
    user_id: str = "unknown_user",
    document_id: str = "unknown_doc",
    session_id: str = "unknown_session"
):
    """Log retrieval experiment metrics & parameters into MLflow."""

    with mlflow.start_run(run_name="retrieval_eval", nested=True):

        # -----------------------------
        # Basic parameters
        # -----------------------------
        mlflow.log_param("question", question)
        mlflow.log_param("chunk_count", len(retrieved_chunks))
        mlflow.log_param("user_id", user_id)
        mlflow.log_param("document_id", document_id)
        mlflow.log_param("session_id", session_id)

        # -----------------------------
        # Metrics for dashboard
        # -----------------------------
        top1_score = retrieved_chunks[0]["score"] if retrieved_chunks else 0
        avg_score = (
            sum(c["score"] for c in retrieved_chunks) / len(retrieved_chunks)
            if retrieved_chunks else 0
        )

        mlflow.log_metric("retrieval_time_ms", retrieval_time_ms)
        mlflow.log_metric("retrieval_score_top1", top1_score)
        mlflow.log_metric("retrieval_score_avg", avg_score)
        mlflow.log_metric("retrieval_chunk_count", len(retrieved_chunks))

        # -----------------------------
        # Log chunks as JSON artifact
        # -----------------------------
        mlflow.log_dict(
            {"retrieved": retrieved_chunks},
            artifact_file="retrieved_chunks.json"
        )
