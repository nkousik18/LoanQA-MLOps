# app/chat_memory.py
from typing import Dict, Any
from app.memory import memory_store
from app.retriever import retriever
from app.prompt_router import build_prompt
from app.backend import generate
from app.output_validation import OutputValidator
import logging

logger = logging.getLogger(__name__)
validator = OutputValidator()


async def chat_memory_route(payload: Dict[str, Any]):
    """
    Handles ChatGPT-style conversational flow + RAG + intent routing.
    Includes per-user memory + memory trimming.
    """

    user_id = payload.get("user_id", "default_user")
    messages = payload.get("messages", [])

    if not messages:
        return {"error": "messages list required"}

    # Extract last user message
    last_user_msg = messages[-1]["content"]

    # ---- Save user message to memory ----
    memory_store.add_turn(user_id, "user", last_user_msg)

    # ---- Retrieve existing memory ----
    history = memory_store.get_history(user_id)
    summary = memory_store.get_summary(user_id)

    # Build readable history block
    formatted_history = summary + "\n"
    for item in history[:-1]:
        formatted_history += f"[{item['role'].upper()}] {item['content']}\n"

    logger.info(f"[ChatMemory] History turns={len(history)} summary_len={len(summary)}")

    # ---- RAG Retrieval ----
    retrieved_chunks = retriever.retrieve(last_user_msg, top_k=3)
    rag_context = "\n\n".join([c["text"] for c in retrieved_chunks])

    # ---- Intent detection + prompt building ----
    prompt, mode, conf = build_prompt(
        question=last_user_msg,
        context_docs=rag_context,
        retrieved_chunks=retrieved_chunks
    )

    # Combine everything → the final LLM prompt
    full_prompt = f"""
# CHAT SUMMARY
{summary}

# CONVERSATION HISTORY
{formatted_history}

# USER QUESTION
{last_user_msg}

# SYSTEM INSTRUCTIONS (intent-based prompt)
{prompt}
"""

    logger.info(f"[ChatMemory] Final prompt length={len(full_prompt)}")

    # ---- Query vLLM LLM ----
    response_text = await generate(full_prompt)

    # ---- Save assistant message to memory ----
    memory_store.add_turn(user_id, "assistant", response_text)

    # ---- Validate answer ----
    validation = validator.validate(response_text, mode, rag_context, retrieved_chunks)
    if not validation["valid"]:
        response_text = validator.rewrite(response_text, mode, rag_context)

    # ---- Final Response ----
    return {
        "mode": mode,
        "router_confidence": conf,
        "response": response_text,
        "retrieved_chunks": retrieved_chunks,
        "memory_turns": len(history),
    }
