import mlflow
import json

EXPERIMENT_NAME = "LoanDocAI_Production"

def log_batch_eval(results):
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        mlflow.set_tag("component", "batch_eval")

        mlflow.log_metric("num_queries", len(results))
        mlflow.log_text(json.dumps(results, indent=2), "batch_results.json")
