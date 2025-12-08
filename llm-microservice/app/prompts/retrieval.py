from typing import List, Dict
from app.guardrails import Guardrail

guard = Guardrail()

def build_retrieval_prompt(
    question: str,
    context: str = "",
    retrieved_chunks: List[Dict] = None,
) -> str:

    safe_query = guard.sanitize_user_question(question)

    return f"""
You are an information retrieval assistant.

{guard.build_global_instructions()}

Task:
- Return the MOST relevant information from the retrieved context.
- Prioritize exact phrases and wording from the document.
- If the context does not contain the answer, say:
  "The retrieved context does not include the requested information."

[USER QUERY]
{safe_query}

[RETRIEVED CONTEXT]
{context}

[ANSWER BASED STRICTLY ON CONTEXT]
""".strip()
