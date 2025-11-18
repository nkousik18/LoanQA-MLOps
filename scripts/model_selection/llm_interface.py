"""
scripts/model_selection/llm_interface.py
Robust Ollama interface supporting all response formats:
- response
- message.content
- output
- done streams
"""

import time
import json
import requests
from scripts.model_selection.logger import log_event


class LLMResponse:
    def __init__(self, text, latency, model_name, success=True, error_msg=None):
        self.text = text
        self.latency = latency
        self.model_name = model_name
        self.success = success
        self.error_msg = error_msg


def generate_response(model_name, prompt, max_retries=3):
    start = time.time()
    url = "http://localhost:11434/api/generate"

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": True
    }

    full_output = ""

    try:
        with requests.post(url, json=payload, stream=True, timeout=300) as resp:
            resp.raise_for_status()

            for raw in resp.iter_lines():
                if not raw:
                    continue

                try:
                    data = json.loads(raw)
                except:
                    continue

                # Try all possible keys returned by Ollama
                if "response" in data:
                    chunk = data["response"]

                elif "message" in data and "content" in data["message"]:
                    chunk = data["message"]["content"]

                elif "output" in data:
                    chunk = data["output"]

                else:
                    chunk = ""

                full_output += chunk

    except Exception as e:
        latency = time.time() - start
        return LLMResponse("", latency, model_name, success=False, error_msg=str(e))

    latency = time.time() - start

    log_event("llm", {
        "model": model_name,
        "latency": latency,
        "output_len": len(full_output)
    })

    return LLMResponse(full_output.strip(), latency, model_name, success=True)
