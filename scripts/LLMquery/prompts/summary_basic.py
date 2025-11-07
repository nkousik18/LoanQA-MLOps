def summary_prompt(text: str) -> str:
    return f"""
You are LoanDocQA+, an intelligent financial document assistant.
Summarize the following text clearly and concisely in 2–3 sentences.
Focus only on the key idea, without adding new details.

Text:
{text}

Summary:
""".strip()
