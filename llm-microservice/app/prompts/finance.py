from typing import List, Dict
from app.guardrails import Guardrail

guard = Guardrail()

def build_finance_prompt(
    question: str,
    context: str = "",
    retrieved_chunks: List[Dict] = None,
) -> str:

    safe_query = guard.sanitize_user_question(question)

    return f"""
You are a financial reasoning assistant.

{guard.build_global_instructions()}

Task:
- Provide a correct financial explanation.
- Use retrieved context if available.
- If context contains relevant definitions, use them.
- Do NOT invent numbers, percentages, fees, or loan terms.

[FINANCE QUESTION]
{safe_query}

[RETRIEVED CONTEXT]
{context}

[ANSWER]
""".strip()
