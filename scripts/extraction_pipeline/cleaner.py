import re

def clean_text(text: str):
    """
    Clean extracted text generically — no domain hardcoding.
    Removes headers, repeated lines, and non-text artifacts.
    """
    lines = text.splitlines()
    cleaned_lines = []
    previous = None
    for line in lines:
        l = line.strip()
        # Skip empty or repetitive lines
        if not l or l == previous:
            continue
        # Remove lines with mostly symbols
        if len(re.sub(r"[A-Za-z0-9]", "", l)) / max(len(l), 1) > 0.5:
            continue
        # Skip lines with common header/footer patterns
        if re.match(r"^page\s*\d+", l, re.I):
            continue
        cleaned_lines.append(l)
        previous = l
    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()
