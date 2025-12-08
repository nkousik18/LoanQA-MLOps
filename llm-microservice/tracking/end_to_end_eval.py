# tracking/end_to_end_eval.py
import mlflow
from typing import Dict, Any, List


def run_end_to_end_experiment(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    intent: str,
    router_conf: float,
    prompt: str,
    llm_output: str,
    timing: Dict[str, float],
    user_id: str = "unknown_user",
    document_id: str = "unknown_doc",
    session_id: str = "unknown_session"
):
    """Full pipeline tracking: retrieval → router → LLM."""

    with mlflow.start_run(run_name="end_to_end_eval", nested=True):

        # -----------------------------
        # Tags for high-level analysis
        # -----------------------------
        mlflow.set_tag("intent", intent)
        mlflow.set_tag("user_id", user_id)
        mlflow.set_tag("document_id", document_id)
        mlflow.set_tag("session_id", session_id)

        # -----------------------------
        # Params that describe the run
        # -----------------------------
        mlflow.log_param("question", question)
        mlflow.log_param("retrieved_count", len(retrieved_chunks))
        mlflow.log_param("router_mode", intent)

        # -----------------------------
        # End-to-end metrics
        # -----------------------------
        mlflow.log_metric("total_time_ms", timing["total_ms"])
        mlflow.log_metric("retrieval_time_ms", timing["retrieval_ms"])
        mlflow.log_metric("router_time_ms", timing["router_ms"])
        mlflow.log_metric("llm_time_ms", timing["llm_ms"])
        mlflow.log_metric("router_confidence", router_conf)

        # Retrieval-specific metrics
        top1_score = retrieved_chunks[0]["score"] if retrieved_chunks else 0
        mlflow.log_metric("retrieval_score_top1", top1_score)

        # -----------------------------
        # Artifacts
        # -----------------------------
        mlflow.log_text(prompt, "final_prompt_used.txt")
        mlflow.log_text(llm_output, "final_llm_output.txt")
        mlflow.log_dict({"chunks": retrieved_chunks}, "all_chunks.json")
        mlflow.log_dict({"timing": timing}, "timing.json")
