# tracking/llm_eval.py
import mlflow


def run_llm_experiment(
    prompt: str,
    llm_output: str,
    validation: dict,
    llm_time_ms: float,
    user_id: str = "unknown_user",
    document_id: str = "unknown_doc",
    session_id: str = "unknown_session"
):
    """Log LLM output, validation results, and latency."""

    with mlflow.start_run(run_name="llm_eval", nested=True):

        # -----------------------------
        # Parameters
        # -----------------------------
        mlflow.log_param("prompt_preview", prompt[:2000])
        mlflow.log_param("user_id", user_id)
        mlflow.log_param("document_id", document_id)
        mlflow.log_param("session_id", session_id)

        # -----------------------------
        # Metrics
        # -----------------------------
        validation_passed = 1 if validation.get("valid", False) else 0

        mlflow.log_metric("llm_latency_ms", llm_time_ms)
        mlflow.log_metric("validation_passed", validation_passed)

        # -----------------------------
        # Artifacts: full prompt/output
        # -----------------------------
        mlflow.log_text(prompt, "full_prompt.txt")
        mlflow.log_text(llm_output, "llm_output.txt")
        mlflow.log_dict(validation, "validation.json")
