# scripts/model_selection/logger.py

import os
import json
import time
from datetime import datetime


def get_timestamp():
    return datetime.utcnow().isoformat()


def log_event(event_type: str, data: dict, folder="logs/model_selection"):
    """
    Writes structured JSON logs into a subfolder based on event type.
    """
    os.makedirs(f"{folder}/{event_type}", exist_ok=True)

    log_path = f"{folder}/{event_type}/{event_type}.jsonl"

    payload = {
        "timestamp": get_timestamp(),
        "event_type": event_type,
        **data
    }

    with open(log_path, "a") as f:
        f.write(json.dumps(payload) + "\n")
