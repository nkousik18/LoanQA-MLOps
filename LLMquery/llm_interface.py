# LLMquery/llm_interface.py
"""
Updated Ollama interface (LangChain ≥0.3)
"""

from langchain_ollama import OllamaLLM
from langchain.callbacks.base import BaseCallbackHandler


class PrintStreamHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token, **kwargs):
        print(token, end="", flush=True)


def load_llm(model_name="llama3", temperature=0, streaming=False, callbacks=None):
    """
    Loads a local Ollama model (new API).
    """
    if callbacks is None:
        callbacks = []

    # ✅ The new class handles streaming internally — just pass callbacks.
    llm = OllamaLLM(
        model=model_name,
        temperature=temperature,
        callbacks=callbacks if streaming else [],
    )
    return llm
