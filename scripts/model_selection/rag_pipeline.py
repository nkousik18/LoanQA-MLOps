"""
RAG Pipeline for LoanDocQA+
"""

import time
import requests
from scripts.LLMquery.prompts.prompt_router import build_prompt, detect_intent


# ------------------------------------------------------------
# Safe Ollama call
# ------------------------------------------------------------
def ollama_generate(model, prompt, url="http://localhost:11434/api/generate"):
    payload = {"model": model, "prompt": prompt, "stream": False}

    r = requests.post(url, json=payload, timeout=300)
    if r.status_code != 200:
        print("[Ollama] Error:", r.text[:200])
        and_prompt = prompt[:4000]  # fallback
        raise RuntimeError(f"Ollama failed: {r.status_code}")

    data = r.json()
    return data.get("response", "")


# ------------------------------------------------------------
# RAG Pipeline
# ------------------------------------------------------------
class RAGPipeline:

    def __init__(self, index_store, top_k=5):
        self.store = index_store
        self.top_k = top_k

    # --------------------------------------------------------
    # Retrieve chunks
    # --------------------------------------------------------
    def retrieve(self, query):
        return self.store.search(query, k=self.top_k)

    # --------------------------------------------------------
    # Format retrieved context
    # --------------------------------------------------------
    def format_context(self, chunks):
        ctx = "\n\n".join(c["text"] for c in chunks)
        return ctx.strip()

    # --------------------------------------------------------
    # Main RAG call
    # --------------------------------------------------------
    def run(self, model, query, doc_text_for_context_only=""):
        t0 = time.time()

        # 1. Intent → prompt template
        intent, conf, gap = detect_intent(query)

        # 2. Retrieve from vector store
        retrieved = self.retrieve(query)
        context = self.format_context(retrieved)

        # 3. Build prompt
        prompt, _, _, _ = build_prompt(
            question=query,
            docs=[context],  # direct context injection
            mode=None
        )

        # 4. LLM call
        t_llm0 = time.time()
        output = ollama_generate(model, prompt)
        t_llm1 = time.time()

        return {
            "intent": intent,
            "prompt": prompt,
            "llm_output": output,
            "retrieved_chunks": retrieved,
            "context_length": len(context),
            "llm_latency": round(t_llm1 - t_llm0, 4),
            "pipeline_latency": round(time.time() - t0, 4)
        }

    # --------------------------------------------------------
    # Debug helper
    # --------------------------------------------------------
    def debug_prompt(self, query):
        intent, conf, gap = detect_intent(query)
        retrieved = self.retrieve(query)
        context = self.format_context(retrieved)

        prompt, _, _, _ = build_prompt(
            docs=[context], question=query, mode=None
        )

        return prompt, context, retrieved
