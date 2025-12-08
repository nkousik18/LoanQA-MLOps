# app/batch.py
from app.router import route


# app/batch.py

from .router import route

async def batch_route(payload: dict):
    outputs = []

    for item in payload.get("questions", []):
        question_text = item.get("question", "")
        resp = await route({"question": question_text})
        outputs.append(resp)

    return {"responses": outputs}

