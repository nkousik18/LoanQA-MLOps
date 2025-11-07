# src/redact.py
from __future__ import annotations
import re

# Match grouped account/CC-like numbers: 1234-5678-9012 or 1234 5678 9012 [optional 4th group]
ACCOUNT_RE = re.compile(r'(?<!\d)(?:\d{4}[- ]\d{4}[- ]\d{4}(?:[- ]\d{1,4})?)(?!\d)')

# U.S. SSN
SSN_RE = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')

# India PAN (ABCDE1234F)
PAN_RE = re.compile(r'\b[A-Z]{5}\d{4}[A-Z]\b')

# Emails
EMAIL_RE = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')

# Phone numbers like +1-617-555-1212, (617) 555-1212, 617-555-1212
PHONE_RE = re.compile(
    r'(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})'
)

def regex_redact(text: str) -> str:
    """
    Replace obvious sensitive tokens with placeholders.
    Order matters: redact account patterns first so phones/emails don't
    swallow pieces of them.
    """
    s = ACCOUNT_RE.sub("[ACCOUNT]", text)
    s = SSN_RE.sub("[PII]", s)
    s = PAN_RE.sub("[PII]", s)
    s = EMAIL_RE.sub("[PII]", s)
    s = PHONE_RE.sub("[PII]", s)
    return s

def assert_clean(text: str) -> None:
    """
    Assert that no obvious unredacted PII or account numbers remain.
    Should be called AFTER regex_redact().
    """
    if "[ACCOUNT]" in text or "[PII]" in text:
        # These placeholders are OK – they mean redaction happened.
        return

    # If raw PII remains, then raise an error.
    if ACCOUNT_RE.search(text) or SSN_RE.search(text) or PAN_RE.search(text) or EMAIL_RE.search(text):
        raise ValueError("Unredacted sensitive data found in output text.")

