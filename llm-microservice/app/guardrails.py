import re
from typing import Optional


class Guardrail:
    """
    Central guardrail used BEFORE prompt construction.
    Ensures safe inputs, prevents hallucination,
    enforces global safety instructions.
    """

    FALLBACK_EMPTY = "I cannot answer because the question is incomplete."
    FALLBACK_UNSAFE = "I cannot help with unsafe or harmful requests."
    FALLBACK_UNKNOWN = "The provided document does not contain enough information to answer."

    # Simple patterns you can expand later
    UNSAFE_PATTERNS = [
        r"hack", r"bypass", r"illegal", r"fraud", r"exploit"
    ]

    def sanitize_user_question(self, text: Optional[str]) -> str:
        """Validate + clean the question."""
        if not text or not text.strip():
            return self.FALLBACK_EMPTY

        cleaned = text.strip()

        # deny unsafe patterns
        for pattern in self.UNSAFE_PATTERNS:
            if re.search(pattern, cleaned, re.IGNORECASE):
                return self.FALLBACK_UNSAFE

        return cleaned

    def build_global_instructions(self) -> str:
        """Global safety + correctness instructions injected into every prompt."""
        return """
GLOBAL RULES:
- Use ONLY the provided text or retrieved context.
- Do NOT guess, invent facts, or fabricate numbers.
- Do NOT reveal chain-of-thought reasoning.
- If unsure, say: "I cannot determine this from the provided information."
- Keep responses concise and factual.
""".strip()
