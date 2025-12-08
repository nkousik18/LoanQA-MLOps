
# app/backend.py
import requests
from typing import AsyncGenerator, List

from .config import settings
from .utils import log


def vllm_url(path: str) -> str:
    return f"{settings.VLLM_API_URL}{path}"


# ----------------------------------------------------
#  SINGLE COMPLETION
# ----------------------------------------------------
async def generate(prompt: str, max_tokens: int = 300) -> str:
    url = vllm_url("/v1/completions")

    payload = {
        "model": settings.VLLM_MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "top_p": 0.9
    }

    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["text"]
    except Exception as e:
        log.error(f"[vLLM] Completion failed: {e}")
        return "[Backend Error: vLLM not reachable]"


# ----------------------------------------------------
#  CHAT COMPLETIONS
#  (Compatible with vLLM /v1/chat/completions)
# ----------------------------------------------------
async def chat(messages: List[dict], max_tokens: int = 300) -> str:
    url = vllm_url("/v1/chat/completions")

    payload = {
        "model": settings.VLLM_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3
    }

    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        log.error(f"[vLLM] Chat completion failed: {e}")
        return "[Backend Error: vLLM chat endpoint unreachable]"

# ----------------------------------------------------
# STREAMING CHAT COMPLETIONS
# ----------------------------------------------------
async def chat_stream(messages: List[dict], max_tokens: int = 300):
    url = vllm_url("/v1/chat/completions")

    payload = {
        "model": settings.VLLM_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.3
    }

    try:
        with requests.post(url, json=payload, stream=True, timeout=120) as r:
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue

                chunk = line.replace("data: ", "")
                if chunk == "[DONE]":
                    break

                # vLLM chunk format
                yield json.loads(chunk)["choices"][0]["delta"].get("content", "")

    except Exception as e:
        log.error(f"[vLLM] Streaming chat failed: {e}")
        yield "[STREAM ERROR]"

# ----------------------------------------------------
# STREAMING (optional)
# ----------------------------------------------------
async def generate_stream(prompt: str, max_tokens: int = 300) -> AsyncGenerator[str, None]:
    url = vllm_url("/v1/completions")

    payload = {
        "model": settings.VLLM_MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": True
    }

    try:
        with requests.post(url,
                           json=payload,
                           stream=True,
                           timeout=120) as r:

            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                chunk = line.replace("data: ", "")
                if chunk == "[DONE]":
                    break
                yield chunk

    except Exception as e:
        log.error(f"[vLLM] Streaming failed: {e}")
        yield "[Streaming Error]"
