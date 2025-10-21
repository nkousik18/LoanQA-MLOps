from LLMquery.prompts.finance_prompt import finance_prompt
from LLMquery.prompts.summary_prompt import summary_prompt
from LLMquery.prompts.translation_prompt import translation_prompt
from LLMquery.prompts.retrieval_prompt import retrieval_prompt
from LLMquery.prompts.explanation_prompt import explanation_prompt


# 🔹 Internal lightweight intent router
INTENT_KEYWORDS = {
    "summary": ["summary", "summarize", "key points", "main points", "overview"],
    "finance": ["calculate", "interest", "emi", "loan", "repayment", "roi", "npv", "irr", "installment", "rate"],
    "translation": ["translate", "in spanish", "in french", "in hindi", "meaning of", "word for"],
    "explanation": ["explain", "difference between", "define", "meaning", "describe", "why", "how does"],
}


def detect_intent(question: str) -> str:
    """Detects high-level user intent from the question text."""
    q = question.lower()
    best_match = "retrieval"
    max_hits = 0
    for intent, keywords in INTENT_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in q)
        if hits > max_hits:
            best_match, max_hits = intent, hits
    return best_match


def build_prompt(question: str, docs, mode: str = None):
    """
    Dynamically builds the appropriate prompt based on detected intent or user-specified mode.
    No hardcoding of task-specific logic — prompt modules handle their own instruction style.
    """
    context = "\n\n".join([d.page_content for d in docs])

    # ✅ Manual override always wins
    if mode:
        router = {
            "summary": summary_prompt,
            "finance": finance_prompt,
            "translation": translation_prompt,
            "explanation": explanation_prompt,
            "retrieval": retrieval_prompt,
        }
        return router.get(mode, retrieval_prompt)(question, context)

    # ✅ Auto detection
    intent = detect_intent(question)

    if intent == "summary":
        return summary_prompt(question, context)
    elif intent == "finance":
        return finance_prompt(question, context)
    elif intent == "translation":
        return translation_prompt(question, context)
    elif intent == "explanation":
        return explanation_prompt(question, context)
    else:
        return retrieval_prompt(question, context)
