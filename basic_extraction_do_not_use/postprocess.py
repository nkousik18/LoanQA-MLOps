import os
import re
import pandas as pd
from spellchecker import SpellChecker
from glossary_utils import FinancialGlossary

# --- Dynamically resolve the glossary path ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
glossary_path = os.path.join(BASE_DIR, "../data/financial_terms.csv")

# Initialize glossary
glossary = FinancialGlossary(glossary_path)

spell = SpellChecker()

def clean_text(text):
    """Clean and normalize OCR text for financial domain."""
    text = text.replace("€", "₹").replace("$", "₹")
    text = re.sub(r"\s+", " ", text)

    # Normalize domain-specific terms
    text = glossary.normalize_terms(text)

    # Spell correction (basic)
    words = text.split()
    corrected = []
    for w in words:
        if w.isalpha():
            corr = spell.correction(w)
            corrected.append(corr if corr else w)
        else:
            corrected.append(w)
    text = " ".join(corrected).strip()
    return text
