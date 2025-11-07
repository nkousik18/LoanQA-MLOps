import re
import pytest
from src.redact import regex_redact, assert_clean

def test_redact_accounts_and_pii():
    text = "Account 1234-5678-9012, PAN ABCDE1234F, SSN 123-45-6789, Email a@b.com, Phone +1-617-555-1212"
    red = regex_redact(text)
    assert "[ACCOUNT]" in red
    assert "[PII]" in red
    # PAN or bank-like tokens should be replaced as [BANK] or [PII] depending on your patterns
    # sanity: nothing raw remains
    assert "1234-5678-9012" not in red
    assert "123-45-6789" not in red
    assert "a@b.com" not in red

def test_assert_clean_passes_when_no_pii_left():
    clean = "hello [PII] and [ACCOUNT] are placeholders"
    assert_clean(clean)  # should not raise

def test_assert_clean_raises_on_leak():
    leaked = "my email is jane.doe@example.com"
    with pytest.raises(ValueError):
        assert_clean(leaked)


