import time
import logging
import functools
import random
import os
import psutil
import mlflow
import sentry_sdk
from botocore.exceptions import ClientError, BotoCoreError
from google.api_core.exceptions import GoogleAPICallError, RetryError

# Setup Logger
logger = logging.getLogger("Pipeline_Monitor")
logger.setLevel(logging.INFO)

# Initialize Sentry if DSN is provided
if os.getenv("SENTRY_DSN"):
    sentry_sdk.init(dsn=os.getenv("SENTRY_DSN"), traces_sample_rate=1.0)


def monitor_task(task_name, retry_count=3):
    """
    Decorator that:
    1. Tracks execution in MLflow.
    2. Logs Memory & CPU usage (Resource Monitoring).
    3. Retries on Cloud Errors.
    4. Alerts Sentry on Crashing Bugs.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 1. Start MLflow Run (Nested allows grouping under one parent run)
            with mlflow.start_run(run_name=task_name, nested=True):

                # --- Resource Monitoring (Start) ---
                process = psutil.Process(os.getpid())
                mem_before = process.memory_info().rss / 1024 / 1024  # MB
                start_time = time.time()

                attempt = 0
                while attempt < retry_count:
                    try:
                        # 2. Execute the Function
                        result = func(*args, **kwargs)

                        # --- Resource Monitoring (End) ---
                        duration = time.time() - start_time
                        mem_after = process.memory_info().rss / 1024 / 1024  # MB
                        mem_diff = mem_after - mem_before

                        # 3. Log Metrics to MLflow
                        mlflow.log_metric("duration_seconds", duration)
                        mlflow.log_metric("memory_usage_mb", mem_after)
                        mlflow.log_metric("memory_growth_mb", mem_diff)
                        mlflow.log_param("status", "SUCCESS")
                        mlflow.log_param("retries_used", attempt)

                        logger.info(f" [{task_name}] Success in {duration:.2f}s | Mem: {mem_after:.1f}MB")
                        return result

                    except (ClientError, BotoCoreError, GoogleAPICallError, RetryError) as e:
                        # 4. Handle Transient Cloud Errors (Retry)
                        attempt += 1
                        wait_time = 2 ** attempt + random.uniform(0, 1)  # Exponential Backoff

                        logger.warning(f" [{task_name}] Retry {attempt}/{retry_count} due to Cloud Error: {e}")
                        mlflow.log_metric("retry_count", attempt)
                        time.sleep(wait_time)

                    except Exception as e:
                        # 5. Handle Critical Code Bugs (Crash)
                        logger.error(f" [{task_name}] CRITICAL FAILURE: {e}")

                        # Alert Sentry
                        sentry_sdk.capture_exception(e)

                        # Log Failure to MLflow
                        mlflow.log_param("status", "FAILED")
                        mlflow.log_param("error_type", type(e).__name__)
                        mlflow.log_param("error_message", str(e))

                        raise e  # Stop pipeline

                # If we exit the loop, retries were exhausted
                err_msg = f" [{task_name}] Failed after {retry_count} retries."
                logger.error(err_msg)
                mlflow.log_param("status", "MAX_RETRIES_EXCEEDED")
                raise RuntimeError(err_msg)

        return wrapper

    return decorator