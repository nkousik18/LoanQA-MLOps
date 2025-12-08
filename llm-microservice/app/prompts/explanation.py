from typing import List, Dict
from app.guardrails import Guardrail

guard = Guardrail()

def build_explanation_prompt(
    question: str,
    context: str = "",
    retrieved_chunks: List[Dict] = None,
) -> str:

    safe_query = guard.sanitize_user_question(question)

    return f"""
You are a financial explainer assistant.

{guard.build_global_instructions()}

Task:
- Explain the concept in clear, simple language.
- Use retrieved context *only if relevant*.
- If the context does NOT contain the answer, give a general explanation.
- NEVER fabricate document-specific numbers or claims.

[QUESTION]
{safe_query}

[RETRIEVED CONTEXT]
{context}

[EXPLANATION]
""".strip()
