# tracking/router_eval.py
import mlflow


def run_router_experiment(
    question: str,
    predicted_mode: str,
    confidence: float,
    router_time_ms: float,
    user_id: str = "unknown_user",
    document_id: str = "unknown_doc",
    session_id: str = "unknown_session"
):
    """Log router evaluation results."""

    with mlflow.start_run(run_name="router_eval", nested=True):

        # -----------------------------
        # Parameters
        # -----------------------------
        mlflow.log_param("question", question)
        mlflow.log_param("predicted_mode", predicted_mode)
        mlflow.log_param("user_id", user_id)
        mlflow.log_param("document_id", document_id)
        mlflow.log_param("session_id", session_id)

        # -----------------------------
        # Metrics for dashboard
        # -----------------------------
        mlflow.log_metric("router_confidence", confidence)
        mlflow.log_metric("router_time_ms", router_time_ms)
