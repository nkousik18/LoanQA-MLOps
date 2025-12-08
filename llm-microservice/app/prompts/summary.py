from typing import List, Dict
from app.guardrails import Guardrail

guard = Guardrail()

def build_summary_prompt(
    question: str,
    context: str = "",
    retrieved_chunks: List[Dict] = None,
) -> str:

    safe_query = guard.sanitize_user_question(question)

    return f"""
You are a document summarization assistant.

{guard.build_global_instructions()}

Task:
- Produce a concise summary using ONLY the retrieved context.
- Do NOT introduce new information.
- Highlight the key ideas exactly as stated in the text.
- If the context is empty, say: "No document context was provided."

[SUMMARY REQUEST]
{safe_query}

[DOCUMENT CONTEXT]
{context}

[SUMMARY]
""".strip()
