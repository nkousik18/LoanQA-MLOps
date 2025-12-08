import re
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class OutputValidator:

    def validate_structure(self, output: str, mode: str) -> bool:
        """Enforce structural rules based on output mode."""
        bad_cot = ["thinking", "thought process", "let me think", "step-by-step"]
        if any(b in output.lower() for b in bad_cot):
            return False

        if mode == "summary" and len(output.split()) > 150:
            return False

        if mode == "translation":
            if "explanation" in output.lower() or "summary" in output.lower():
                return False

        return True

    def hallucination_numbers(self, output: str, context: str) -> bool:
        """Detect numeric hallucination: numbers in answer but absent from context."""
        out_nums = set(re.findall(r"\b\d+\.?\d*\b", output))
        ctx_nums = set(re.findall(r"\b\d+\.?\d*\b", context))
        return len(out_nums - ctx_nums) > 0

    def hallucination_entities(self, output: str, context: str) -> bool:
        """Optional — detect fabricated terms not in context."""
        # Simple heuristic: unique long words not in context
        out_words = {w.lower() for w in output.split() if len(w) > 6}
        ctx_words = {w.lower() for w in context.split() if len(w) > 6}
        return len(out_words - ctx_words) > 10  # threshold

    def validate_safety(self, output: str) -> bool:
        forbidden = [
            "legal advice",
            "financial advice",
            "my recommendation",
            "personally I suggest",
        ]
        return not any(f in output.lower() for f in forbidden)

    def citation_check(self, output: str, selected_chunks: List[Dict]) -> bool:
        """Ensure answer references chunk numbers if retrieval mode."""
        if not selected_chunks:
            return True

        # Must contain at least one chunk reference
        return any(f"[chunk {i}]" in output.lower() for i in range(1, len(selected_chunks) + 1))

    def validate(self, output: str, mode: str, context: str, chunks: List[Dict]):
        """Run full validation pipeline."""
        result = {"valid": True, "issues": []}

        if not self.validate_structure(output, mode):
            result["valid"] = False
            result["issues"].append("Structure violation")

        if self.hallucination_numbers(output, context):
            result["valid"] = False
            result["issues"].append("Potential numeric hallucination")

        if not self.validate_safety(output):
            result["valid"] = False
            result["issues"].append("Safety violation")

        if mode == "retrieval" and not self.citation_check(output, chunks):
            result["valid"] = False
            result["issues"].append("Missing required citations")

        return result

    # -------------------------------
    # AUTO-REWRITE
    # -------------------------------
    def rewrite(self, output: str, mode: str, context: str) -> str:
        """Rewrite the invalid answer into a safe one."""
        if mode == "retrieval":
            return "The answer is not fully supported by the retrieved document. Based on the context provided, here are the facts:\n\n" + context

        if mode == "summary":
            return "Here is a corrected summary based strictly on the provided text:\n" + context[:300]

        if mode == "translation":
            return "Unable to validate translation. Here is a safe literal translation:\n" + context

        # Default fallback
        return "The model attempted to answer, but the output violated safety or factuality guidelines. Based on the document provided:\n\n" + context
