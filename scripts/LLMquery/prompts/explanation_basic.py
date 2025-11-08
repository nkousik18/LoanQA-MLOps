def explanation_prompt(text: str) -> str:
    return f"""
You are LoanDocQA+, an expert at explaining complex legal and financial terms.
Explain the following text in simple, clear, and professional language.
Avoid technical jargon, but preserve key meaning.

Text:
{text}

Explanation:
""".strip()
