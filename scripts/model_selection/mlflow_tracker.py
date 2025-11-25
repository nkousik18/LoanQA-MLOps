"""
MLflow tracking utilities for LoanQA RAG evaluation
"""

import mlflow
import mlflow.sklearn
from typing import Dict, Any
import pandas as pd
import os


class MLflowRAGTracker:
    """
    Tracks RAG pipeline experiments with MLflow
    """

    def __init__(self, experiment_name="LoanQA_RAG_Evaluation"):
        """
        Initialize MLflow experiment with tracking in logs folder

        Args:
            experiment_name: Name of the MLflow experiment
        """
        # Set tracking URI to logs folder
        tracking_uri = "./logs/mlruns"
        os.makedirs(tracking_uri, exist_ok=True)

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.run_id = None

        print(f"[MLflow] Tracking URI: {tracking_uri}")
        print(f"[MLflow] Experiment: {experiment_name}")

    def start_run(self, run_name: str, model_name: str, mode: str):
        """Start a new MLflow run"""
        self.run = mlflow.start_run(run_name=run_name)
        self.run_id = self.run.info.run_id

        # Log initial parameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("evaluation_mode", mode)

        print(f"[MLflow] Started run: {run_name}")
        print(f"[MLflow] Run ID: {self.run_id}")

        return self.run

    def log_query_result(self,
                         doc_id: str,
                         intent: str,
                         query: str,
                         metrics: Dict[str, float],
                         latency: Dict[str, float]):
        """
        Log individual query evaluation metrics
        """
        # Log metrics with prefixes for organization
        mlflow.log_metric(f"{intent}_groundedness", metrics["groundedness"])
        mlflow.log_metric(f"{intent}_severity", metrics["severity"])
        mlflow.log_metric(f"{intent}_confidence", metrics["confidence"])
        mlflow.log_metric(f"{intent}_divergence", metrics["summary_divergence"])

        # Log latency
        mlflow.log_metric(f"{intent}_llm_latency", latency["llm_latency"])
        mlflow.log_metric(f"{intent}_pipeline_latency", latency["pipeline_latency"])

        # Log hallucination flag as metric (1 or 0)
        mlflow.log_metric(f"{intent}_hallucinated", 1.0 if metrics["hallucinated"] else 0.0)

    def log_aggregate_metrics(self, df: pd.DataFrame):
        """
        Log aggregate metrics across all queries
        """
        # Overall averages
        mlflow.log_metric("avg_groundedness", df["groundedness"].mean())
        mlflow.log_metric("avg_severity", df["severity"].mean())
        mlflow.log_metric("avg_confidence", df["confidence"].mean())
        mlflow.log_metric("avg_divergence", df["summary_divergence"].mean())

        # Hallucination rate
        hallucination_rate = df["hallucinated"].sum() / len(df)
        mlflow.log_metric("hallucination_rate", hallucination_rate)

        # Average latency
        mlflow.log_metric("avg_llm_latency", df["llm_latency"].mean())
        mlflow.log_metric("avg_pipeline_latency", df["pipeline_latency"].mean())

        # Per-intent metrics
        for intent in df["intent_group"].unique():
            intent_df = df[df["intent_group"] == intent]
            mlflow.log_metric(f"{intent}_avg_groundedness", intent_df["groundedness"].mean())
            mlflow.log_metric(f"{intent}_hallucination_rate",
                              intent_df["hallucinated"].sum() / len(intent_df))

        print(f"[MLflow] Logged aggregate metrics")

    def log_evaluation_summary(self, csv_path: str):
        """Log the evaluation CSV as artifact"""
        mlflow.log_artifact(csv_path, "evaluation_results")
        print(f"[MLflow] Logged evaluation summary: {csv_path}")

    def log_parameters(self, params: Dict[str, Any]):
        """Log additional parameters"""
        for key, value in params.items():
            mlflow.log_param(key, value)

    def end_run(self):
        """End the current MLflow run"""
        if self.run:
            mlflow.end_run()
            print(f"[MLflow] Ended run: {self.run_id}")
            print(f"[MLflow] View at: http://localhost:5000")


def compare_models(experiment_name="LoanQA_RAG_Evaluation"):
    """
    Helper function to compare all models in the experiment
    Returns a DataFrame with model comparison
    """
    from mlflow.tracking import MlflowClient

    # Set tracking URI
    mlflow.set_tracking_uri("./logs/mlruns")

    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if not experiment:
        print(f"No experiment found: {experiment_name}")
        return None

    runs = client.search_runs(experiment.experiment_id)

    comparison_data = []
    for run in runs:
        comparison_data.append({
            "run_id": run.info.run_id,
            "model_name": run.data.params.get("model_name", "N/A"),
            "mode": run.data.params.get("evaluation_mode", "N/A"),
            "avg_groundedness": run.data.metrics.get("avg_groundedness", 0),
            "hallucination_rate": run.data.metrics.get("hallucination_rate", 0),
            "avg_latency": run.data.metrics.get("avg_pipeline_latency", 0)
        })

    return pd.DataFrame(comparison_data)