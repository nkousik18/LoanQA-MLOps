"""
llm_executor.py
────────────────────────────────────────────
Executes and streams LLM prompts using a local Ollama model.

Optimized for local use:
- No OpenAI dependency
- Minimal latency
- Real-time streaming with yield
- Structured logging
"""

import os
import datetime
import logging
import json  # ✅ Needed for streaming JSON parsing
import requests  # Local Ollama API calls

# ============================================================
# Logging setup
# ============================================================
LOG_DIR = "logs/llm_logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, f"llm_execution_{datetime.date.today()}.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ============================================================
# Model configuration (Local Ollama only)
# ============================================================
LLM_BACKEND = "ollama"
OLLAMA_MODEL = str(os.getenv("OLLAMA_MODEL", "phi3")).strip()  # Default model
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")


# ============================================================
# Core synchronous execution
# ============================================================

def run_llm(prompt: str) -> str:
    """
    Executes a given prompt using the local Ollama backend
    and returns the full generated response as a string.
    """
    try:
        response = _run_ollama(prompt)
        logging.info(f"[SUCCESS] LLM prompt executed successfully: {prompt[:80]}...")
        return response

    except Exception as e:
        logging.error(f"[ERROR] LLM execution failed: {str(e)}")
        raise RuntimeError(f"LLM execution failed: {e}")


# ============================================================
# Streaming execution
# ============================================================

def stream_llm(prompt: str):
    """
    Streams an LLM response from Ollama to the client in real time.
    Cleans partial tokens and collapses spacing for readable output.
    """
    if not prompt or not prompt.strip():
        yield "[ERROR] Empty prompt."
        return

    payload = {
        "model": OLLAMA_MODEL.strip(),
        "prompt": prompt.strip(),
        "stream": True
    }

    import re

    try:
        with requests.post(OLLAMA_API_URL, json=payload, stream=True) as resp:
            resp.raise_for_status()
            buffer = ""

            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    token = data.get("response", "")
                    if token:
                        # 🔹 Merge token smoothly with previous partials
                        buffer += token
                        # Replace multiple spaces, fix partial word splits
                        clean = re.sub(r"\s{2,}", " ", buffer)
                        # Only yield complete sentences or line breaks periodically
                        if any(c in clean for c in [".", "\n", "!", "?"]) or len(clean) > 80:
                            yield clean.strip()
                            buffer = ""
                    if data.get("done", False):
                        if buffer.strip():
                            yield buffer.strip()
                        break
                except json.JSONDecodeError:
                    continue

    except Exception as e:
        msg = f"[Streaming error: {e}]"
        logging.error(msg)
        yield msg



# ============================================================
# Non-streaming Ollama call (used by run_llm)
# ============================================================

def _run_ollama(prompt: str) -> str:
    """
    Runs a prompt using the Ollama local model (phi3, mistral, llama3).
    Matches the tested /api/generate structure.
    """

    # Safety check
    if not prompt or not isinstance(prompt, str):
        raise ValueError("Prompt must be a non-empty string.")

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt.strip(),
        "stream": False
    }

    try:
        response = requests.post(
            OLLAMA_API_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=180  # prevent indefinite hangs
        )

        response.raise_for_status()
        result = response.json()

        # Extract response text
        output = result.get("response", "").strip()
        if not output:
            raise RuntimeError("No response text returned from Ollama.")

        return output

    except requests.exceptions.RequestException as e:
        logging.error(f"[Ollama Error] HTTP Request failed: {e}")
        raise RuntimeError(f"Ollama request failed: {e}")
    except Exception as e:
        logging.error(f"[Ollama Error] {e}")
        raise RuntimeError(f"Ollama call failed: {e}")
