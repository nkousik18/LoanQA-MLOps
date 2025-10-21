import re

def postprocess_text(text: str):
    """Normalize bullets, spacing, and capitalization."""
    text = re.sub(r"[\t|•@#_=~]", "", text)
    text = re.sub(r"\n[-–—]\s*", "\n• ", text)
    text = re.sub(r"(?<!\.)\n(?=[a-z])", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)  # separate glued words
    return text.strip()
