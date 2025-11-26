import os
from datetime import datetime

def log_pipeline_completion():
    """
    Logs the completion of the OCR pipeline into logs/pipeline_run_log.txt
    """
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", "pipeline_run_log.txt")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f"[{timestamp}] ✅ OCR pipeline completed successfully.\n"
    
    with open(log_path, "a") as f:
        f.write(message)
    
    print(f"📘 Pipeline completion logged at {log_path}")
