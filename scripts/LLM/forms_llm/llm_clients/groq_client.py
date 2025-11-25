import os
from groq import Groq

# ✅ only change: model
GROQ_MODEL = "llama-3.1-8b-instant"


def get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable is not set.")
    return Groq(api_key=api_key)


def call_groq_chat(system_prompt: str, user_prompt: str) -> str:
    client = get_groq_client()

    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        # ✅ only change: tuning (demo/testing stable + enough output)
        temperature=0.05,
        max_tokens=1400,
    )

    return resp.choices[0].message.content.strip()
