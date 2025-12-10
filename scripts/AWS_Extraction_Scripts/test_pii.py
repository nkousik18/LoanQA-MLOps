# import unittest
# import sys
# import os
#
# # ---------------------------------------------------------------------
# # PATH SETUP
# # ---------------------------------------------------------------------
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.abspath(os.path.join(current_dir, "../../"))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)
#
# from pii_masking import PIIMasker
#
#
# class TestPIIMasking(unittest.TestCase):
#     def setUp(self):
#         self.masker = PIIMasker()
#
#     def test_ssn_masking(self):
#         print("\n--- Testing SSN Masking ---")
#         input_text = "My SSN is 123-45-6789."
#         masked_output = self.masker.mask_text(input_text)
#         print(f"Output: {masked_output}")
#         self.assertEqual(masked_output, "My SSN is <SSN_REDACTED>.")
#
#     def test_email_masking(self):
#         print("\n--- Testing Email Masking ---")
#         input_text = "Contact: john.doe@example.com."
#         masked_output = self.masker.mask_text(input_text)
#         print(f"Output: {masked_output}")
#         self.assertEqual(masked_output, "Contact: <EMAIL_REDACTED>.")
#
#     def test_bank_account_masking(self):
#         print("\n--- Testing Bank Account Masking ---")
#         # 12 digits -> Bank Account (should be masked)
#         input_text = "Wire to account 123456789012 please."
#         masked_output = self.masker.mask_text(input_text)
#         print(f"Output: {masked_output}")
#         self.assertEqual(masked_output, "Wire to account <BANK_ACC_REDACTED> please.")
#
#     def test_financial_preservation(self):
#         print("\n--- Testing Financial Data Preservation ---")
#         # 5 digits -> Money/Zip (should NOT be masked)
#         input_text = "Loan of $10000 at 5.5% interest."
#         masked_output = self.masker.mask_text(input_text)
#         print(f"Output: {masked_output}")
#         self.assertEqual(masked_output, input_text)
#
#     def test_full_loan_agreement(self):
#         print("\n--- Testing Full Loan Agreement (Integration Test) ---")
#
#         # The exact text provided
#         raw_document_text = """
# === PAGE 1 ===
# PERSONAL LOAN AGREEMENT
# 1. THE PARTIES. This Personal Loan Agreement ("Agreement") is made as of this
# 10/08/2025
# (mm/dd/yyyy), by and between:
# Borrower: yaswanth kumar reddy
# , with a mailing address of yaswanth.gujjula@gmail.com
# ,
# City of boston
# , State of MA
# ("Borrower"), and
# Lender: BANK
# , with a mailing address of bank.x@gmail.com
# ,
# City of boston
# , State of MA
# ("Lender").
# 2. LOAN AMOUNT. The Lender shall loan to Borrower the amount of
# ten thousand dollars
# Dollars ($ 10000
# ) ("Loan").
# 3. INTEREST.
# The Loan will bear interest at a rate of eight
# Percent (8
# %)
# compounded annually. The rate must be equal to or less than the usury rate in the State
# of the Borrower.
# The Loan will not bear interest.
# 4. PAYMENT. The Loan shall be due and payable, including the principal and any accrued
# interest, in one (1) of the following ways (check one):
# - Borrower will make weekly payments of $
# beginning on
# (mm/dd/yyyy) and to be paid every
# (day of week)
# until the Loan is paid, ending on
# (mm/dd/yyyy) ("Term").
# - Borrower will make monthly payments of $313.36
# beginning on
# 11/08/2025
# (mm/dd/yyyy) and to be paid on the 8th
# of every month until the
# Loan is paid, ending on 10/08/2028
# (mm/dd/yyyy) ("Term").
# - Borrower will make lump sum payment of $
# to be paid on
# (mm/dd/yyyy) ("Term").
# - Other:
# ("Term").
# All payments made by the Borrower are to be applied first to any accrued interest and
# secondly to the principal balance.
# 5. PAYMENT INSTRUCTIONS. The Borrower shall make payment to the Lender in under the
# following instructions:
# All monthly payments of $313 36 shall be made by electronic transfer (ACH/online payment) to the Lender's designated bank account on or before the 8th day of each month, beginning November 8. 2025
# .
# 6. LATE FEE. If any payment is 10
# day(s) late, the Lender shall: (check one)
# - Charge a late fee of 25$
# .
# - Shall not charge a late.
# eSign
# Page 1 of 3
# """
#         # Run the masker
#         masked_output = self.masker.mask_text(raw_document_text)
#
#         print(f"--- MASKED OUTPUT START ---\n{masked_output}\n--- MASKED OUTPUT END ---")
#
#         # ASSERTIONS
#
#         # 1. Check Emails are redacted
#         self.assertIn("<EMAIL_REDACTED>", masked_output)
#         self.assertNotIn("yaswanth.gujjula@gmail.com", masked_output)
#         self.assertNotIn("bank.x@gmail.com", masked_output)
#
#         # 2. Check Financials are PRESERVED
#         # $10000 should NOT be masked as a bank account (it's 5 digits, bank acc regex expects 8-12)
#         self.assertIn("$ 10000", masked_output)
#
#         # Interest rate 8 %
#         self.assertIn("Percent (8\n%)", masked_output)
#
#         # Monthly payment
#         self.assertIn("$313.36", masked_output)
#
#         # Dates (Should not look like SSNs)
#         self.assertIn("10/08/2025", masked_output)
#
#         print("✅ Full Document Test Passed: Emails masked, Financials preserved.")
#
#
# if __name__ == "__main__":
#     unittest.main()

import sys
import os

# ---------------------------------------------------------------------
# PATH SETUP (Boilerplate to find the pii_masking script)
# ---------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pii_masking import PIIMasker


def get_multiline_input():
    """Helper to capture multi-line text (like pasting a PDF)."""
    print("\n" + "=" * 60)
    print("PASTE YOUR TEXT BELOW.")
    print("Type 'END' on a new line and hit Enter to process.")
    print("Type 'EXIT' to quit the program.")
    print("=" * 60 + "\n")

    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break

        if line.strip().upper() == 'END':
            return "\n".join(lines)
        if line.strip().upper() == 'EXIT':
            return "EXIT"

        lines.append(line)

    return "\n".join(lines)


def run_interactive_mode():
    masker = PIIMasker()

    while True:
        user_input = get_multiline_input()

        if user_input == "EXIT":
            print("Exiting...")
            break

        if not user_input.strip():
            print("No text entered. Try again.")
            continue

        # --- RUN THE MASKER ---
        masked_output = masker.mask_text(user_input)

        # --- PRINT RESULTS ---
        print("\n" + "*" * 20 + " MASKED OUTPUT " + "*" * 20)
        print(masked_output)
        print("*" * 55 + "\n")


if __name__ == "__main__":
    run_interactive_mode()