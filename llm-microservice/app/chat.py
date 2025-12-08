# app/chat.py
from app.backend import chat
from app.prompt_router import build_prompt


async def chat_route(payload: dict):
    messages = payload.get("messages", [])

    if not messages:
        return {"error": "messages list cannot be empty"}

    # Last user message → used for RAG
    user_last = messages[-1]["content"]

    # Build hybrid prompt
    prompt, mode, conf = build_prompt(user_last, "", None)

    # Convert our prompt → chat format
    messages.append({"role": "system", "content": prompt})

    llm_response = await chat(messages)

    return {
        "response": llm_response,
        "mode": mode,
        "confidence": conf,
    }
