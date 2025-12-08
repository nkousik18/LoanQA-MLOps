from typing import List, Dict
from app.guardrails import Guardrail

guard = Guardrail()

def build_translation_prompt(
    question: str,
    context: str = "",
    retrieved_chunks: List[Dict] = None,
) -> str:

    safe_query = guard.sanitize_user_question(question)

    return f"""
You are a precise translation assistant.

{guard.build_global_instructions()}

Task:
- Translate the requested text accurately.
- Ignore unrelated context unless it contains the text to be translated.
- Keep meaning faithful; do NOT summarize or explain.

[TRANSLATION REQUEST]
{safe_query}

[OPTIONAL CONTEXT]
{context}

[TRANSLATION]
""".strip()
