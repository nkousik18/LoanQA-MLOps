import os
import sys
import re
from pathlib import Path

from scripts.aws_extraction_scripts.log_utils import get_logger

# ---------------------------------------------------------------------
# Optional: ensure project root is on sys.path (for safety if run directly)
# ---------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))  # .../doc-understand

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

LOGGER = get_logger(__name__)


class PIIMasker:
    """
    Sanitizes text data by masking PII while preserving financial context.

    UPDATES:
    - Fixed "By" false positives (now requires start-of-line).
    - Added International Phone Number support.
    - Added Address Context masking.
    """

    # --- 1. STANDARD PII PATTERNS ---

    # SSN: US Format (XXX-XX-XXXX)
    REGEX_SSN = r"\b(?!000|666|9\d{2})\d{3}[- ]?(?!00)\d{2}[- ]?(?!0000)\d{4}\b"

    # EMAIL: Standard
    REGEX_EMAIL = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"

    # PHONE: International & US
    # Matches: +91 22 6601 6601, 1860 266 6601, (123) 456-7890
    # Logic: Optional Country Code -> Groups of 2-5 digits -> Separators (space/dot/dash)
    REGEX_PHONE = r"(?:\+?\d{1,3}[ -]?)?\(?\d{2,5}\)?[ -]?\d{3,5}[ -]?\d{3,5}\b"

    # CREDIT CARD: 13-19 digits
    REGEX_CREDIT_CARD = r"\b(?:\d{4}[- ]?){3}\d{4}\b"

    # BANK ACCOUNT / AADHAAR: 8-12 Digits
    # Negative Lookbehind: Ensure it's not a dollar amount ($) or decimal (.)
    REGEX_BANK_ACCOUNT = r"(?<!\$)(?<!\.)\b\d{8,12}\b(?!\%)"

    # --- 2. CONTEXTUAL PATTERNS ---

    # NAMES (Standard Anchors)
    REGEX_NAMES_STANDARD = (
        r"(?i)\b(Borrower|Lender|Name|Witness)\s*[:\.]?\s*"
        r"([A-Za-z\s\.]+?)(?=\n|,|\s+with)"
    )

    # NAMES (Strict "By" Anchor)
    REGEX_NAMES_STRICT_BY = r"(?i)(?:^|\n)\s*(By|Signed By)\s*[:\.]?\s*([A-Za-z\s\.]+?)(?=\n)"

    # ADDRESSES
    REGEX_ADDRESS = (
        r"(?i)\b(Address|Residing at)\s*[:\.]?\s*"
        r"([A-Za-z0-9\s,\.\-\/]+?)(?=\n|Tel|Mobile|PIN)"
    )

    def __init__(self):
        # Compile standard patterns
        self.patterns = {
            "SSN": re.compile(self.REGEX_SSN),
            "EMAIL": re.compile(self.REGEX_EMAIL),
            "PHONE": re.compile(self.REGEX_PHONE),
            "CREDIT_CARD": re.compile(self.REGEX_CREDIT_CARD),
            "BANK_ACC": re.compile(self.REGEX_BANK_ACCOUNT),
        }

        # Compile Contextual patterns
        self.name_std_pattern = re.compile(self.REGEX_NAMES_STANDARD)
        self.name_by_pattern = re.compile(self.REGEX_NAMES_STRICT_BY)
        self.address_pattern = re.compile(self.REGEX_ADDRESS)

        LOGGER.debug("[PIIMasker] Initialized PII patterns.")

    def mask_text(self, text: str) -> str:
        """
        Masks PII and Contextual Names/Addresses in a plain string.
        """
        if not text:
            return ""

        masked_text = text

        # 1. Mask Standard PII (SSN, Email, Phone, etc.)
        for label, pattern in self.patterns.items():
            masked_text = pattern.sub(f"<{label}_REDACTED>", masked_text)

        # 2. Mask Address Context
        masked_text = self.address_pattern.sub(
            lambda m: f"{m.group(1)}: <ADDRESS_REDACTED>",
            masked_text,
        )

        # 3. Mask Standard Names (Borrower, Lender, etc.)
        masked_text = self.name_std_pattern.sub(
            lambda m: f"{m.group(1)}: <NAME_REDACTED>",
            masked_text,
        )

        # 4. Mask Strict "By" Names (Signature lines)
        masked_text = self.name_by_pattern.sub(
            lambda m: f"\n{m.group(1)}: <NAME_REDACTED>",
            masked_text,
        )

        return masked_text

    def mask_textract_blocks(self, blocks: list) -> list:
        """
        Iterates through AWS Textract 'Blocks' and masks 'Text'.
        """
        masked_count = 0

        for block in blocks:
            if block.get("BlockType") in ["LINE", "WORD"] and "Text" in block:
                original = block["Text"]
                masked = self.mask_text(original)

                if original != masked:
                    block["Text"] = masked
                    masked_count += 1

        if masked_count > 0:
            LOGGER.info(f"[PIIMasker] Masked PII in {masked_count} Textract blocks.")

        return blocks
