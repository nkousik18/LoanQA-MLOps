# LLMquery/query_engine_grounded.py
"""
Streaming-aware grounded QA engine.
Supports both full-response and progressive token streaming.
"""

import asyncio
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.callbacks.base import AsyncCallbackHandler


# ────────────────────────────────────────────────
# 🧠 1. Normal (non-streaming) chain setup
# ────────────────────────────────────────────────
def create_grounded_qa_chain(llm, vectorstore, k=4):
    """
    Creates a QA chain that includes retrieved source text for transparency.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )
    return chain


# ────────────────────────────────────────────────
# ⚡ 2. Progressive streaming implementation
# ────────────────────────────────────────────────
async def stream_grounded_answer(llm, vectorstore, question: str, k: int = 4):
    """
    Async generator that streams tokens as they are produced by the LLM.
    Yields partial tokens one by one (SSE / websocket compatible).
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # Custom callback handler to stream tokens
    queue: asyncio.Queue[str] = asyncio.Queue()

    class StreamHandler(AsyncCallbackHandler):
        async def on_llm_new_token(self, token: str, **kwargs):
            await queue.put(token)

        async def on_llm_end(self, *args, **kwargs):
            await queue.put("[END]")

    # Attach callback
    callbacks = [StreamHandler()]
    from langchain_ollama import OllamaLLM
    llm_stream = OllamaLLM(model=llm.model, temperature=0, callbacks=callbacks)

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm_stream,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )

    # Launch the LLM chain in background
    async def run_chain():
        await chain.acall({"question": question})

    task = asyncio.create_task(run_chain())

    # Yield tokens as they arrive
    while True:
        token = await queue.get()
        if token == "[END]":
            break
        yield token

    await task


# ────────────────────────────────────────────────
# 🧩 3. Regular blocking helper (for JSON endpoint)
# ────────────────────────────────────────────────
def ask_grounded_question(chain, query: str):
    """
    Synchronous version used by /query endpoint.
    Waits for the complete answer before returning.
    """
    response = chain({"question": query})
    answer = response["answer"]
    sources = response.get("sources", "")

    if "source_documents" in response:
        context = "\n---\n".join([doc.page_content for doc in response["source_documents"]])
    else:
        context = "No document context found."

    return answer, sources, context
