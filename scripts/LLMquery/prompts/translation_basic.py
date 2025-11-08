def translation_prompt(text: str, lang: str = "en") -> str:
    return f"""
You are LoanDocQA+, a multilingual translation assistant.
Translate the following English text into {lang} accurately and professionally.

Text:
{text}

Translation:
""".strip()
