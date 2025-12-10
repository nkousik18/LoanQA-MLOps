# import re
# import logging
#
# logger = logging.getLogger("pii_masker")
#
#
# class PIIMasker:
#     """
#     Sanitizes text data by masking PII while preserving financial context.
#     Designed for Loan Documents (preserves rates, dates, and currency).
#     """
#
#     # 1. SSN: Strict XXX-XX-XXXX format
#     # Avoids masking timestamps or other hyphenated codes
#     REGEX_SSN = r"\b(?!000|666|9\d{2})\d{3}[- ]?(?!00)\d{2}[- ]?(?!0000)\d{4}\b"
#
#     # 2. EMAIL: Standard email regex
#     REGEX_EMAIL = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
#
#     # 3. PHONE: US Standard (123-456-7890) or (123) 456-7890
#     REGEX_PHONE = r"\b(?:\+?1[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b"
#
#     # 4. CREDIT CARD: 13-19 digits, often grouped
#     REGEX_CREDIT_CARD = r"\b(?:\d{4}[- ]?){3}\d{4}\b"
#
#     # 5. BANK ACCOUNT (The Tricky One)
#     # We match 8-12 digits, BUT we use look-behinds/look-aheads to ensure
#     # it is NOT a dollar amount ($), a percentage (%), or a decimal (float).
#     # Matches: "Account 12345678"
#     # Ignores: "$12345678", "12345678%", "0.12345678"
#     REGEX_BANK_ACCOUNT = r"(?<!\$)(?<!\.)\b\d{8,12}\b(?!\%)"
#
#     def __init__(self):
#         # Compile patterns for speed
#         self.patterns = {
#             "SSN": re.compile(self.REGEX_SSN),
#             "EMAIL": re.compile(self.REGEX_EMAIL),
#             "PHONE": re.compile(self.REGEX_PHONE),
#             "CREDIT_CARD": re.compile(self.REGEX_CREDIT_CARD),
#             "BANK_ACC": re.compile(self.REGEX_BANK_ACCOUNT),
#         }
#
#     def mask_text(self, text: str) -> str:
#         """
#         Masks PII in a plain string.
#         Returns: "My SSN is <SSN_REDACTED>."
#         """
#         if not text:
#             return ""
#
#         masked_text = text
#         for label, pattern in self.patterns.items():
#             masked_text = pattern.sub(f"<{label}_REDACTED>", masked_text)
#
#         return masked_text
#
#     def mask_textract_blocks(self, blocks: list) -> list:
#         """
#         Iterates through AWS Textract 'Blocks' JSON structure and masks
#         the 'Text' field in every LINE and WORD block.
#         """
#         # We assume 'blocks' is a mutable list of dicts.
#         # We modify it in place to save memory.
#         masked_count = 0
#
#         for block in blocks:
#             if block.get("BlockType") in ["LINE", "WORD"] and "Text" in block:
#                 original = block["Text"]
#                 masked = self.mask_text(original)
#
#                 if original != masked:
#                     block["Text"] = masked
#                     masked_count += 1
#
#         if masked_count > 0:
#             logger.info(f"Masked PII in {masked_count} Textract blocks.")
#
#         return blocks

import re
import logging

logger = logging.getLogger("pii_masker")


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
    # Looks for "Borrower:", "Lender:", "Name:", "Witness:" anywhere
    REGEX_NAMES_STANDARD = r"(?i)\b(Borrower|Lender|Name|Witness)\s*[:\.]?\s*([A-Za-z\s\.]+?)(?=\n|,|\s+with)"

    # NAMES (Strict "By" Anchor)
    # Only matches "By" or "Signed By" if it starts a new line (to avoid 'issued by...')
    REGEX_NAMES_STRICT_BY = r"(?i)(?:^|\n)\s*(By|Signed By)\s*[:\.]?\s*([A-Za-z\s\.]+?)(?=\n)"

    # ADDRESSES
    # Looks for "Address", "Residing at" until it hits a newline or a Keyword (Tel, Mobile, PIN)
    REGEX_ADDRESS = r"(?i)\b(Address|Residing at)\s*[:\.]?\s*([A-Za-z0-9\s,\.\-\/]+?)(?=\n|Tel|Mobile|PIN)"

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
        # "Address: 123 Main St" -> "Address: <ADDRESS_REDACTED>"
        masked_text = self.address_pattern.sub(
            lambda m: f"{m.group(1)}: <ADDRESS_REDACTED>",
            masked_text
        )

        # 3. Mask Standard Names (Borrower, Lender, etc.)
        masked_text = self.name_std_pattern.sub(
            lambda m: f"{m.group(1)}: <NAME_REDACTED>",
            masked_text
        )

        # 4. Mask Strict "By" Names (Signature lines)
        # Note: We reconstruct the newline in the replacement to keep formatting
        masked_text = self.name_by_pattern.sub(
            lambda m: f"\n{m.group(1)}: <NAME_REDACTED>",
            masked_text
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
            logger.info(f"Masked PII in {masked_count} Textract blocks.")

        return blocks