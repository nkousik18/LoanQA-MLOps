import re
from spellchecker import SpellChecker
from glossary_utils import FinancialGlossary

spell = SpellChecker()
glossary = FinancialGlossary("data/financial_terms.csv")  # path to your glossary

def clean_text(text):
    text = text.replace("€", "₹").replace("$", "₹")
    text = re.sub(r"\s+", " ", text)

    # 1. Domain-aware corrections
    text = glossary.normalize_terms(text)

    # 2. General spell correction
    words = text.split()
    corrected = []
    for w in words:
        if w.isalpha():
            corr = spell.correction(w)
            corrected.append(corr if corr is not None else w)
        else:
            corrected.append(w)

    # 3. Final normalization
    text = " ".join(corrected).strip()
    return text
