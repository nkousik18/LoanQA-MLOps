import pandas as pd
import re
import json

class FinancialGlossary:
    def __init__(self, glossary_path=None, finrad_csv=None):
        """
        glossary_path: path to your main glossary (term + definition)
        finrad_csv: optional path to FinRAD dataset (extra terms or definitions)
        """
        self.glossary = {}

        if glossary_path:
            self._load_base(glossary_path)
        if finrad_csv:
            self._extend_from_finrad(finrad_csv)

        # Convert to lowercase for consistent matching
        self.term_map = {term.lower(): definition for term, definition in self.glossary.items()}

        # Known ambiguous single-word terms — not deleted, but filtered by context later
        self.ambiguous_terms = {
            "cover", "option", "offer", "life", "benefits", "labor", "money", "time",
            "note", "rate", "total", "will", "information", "y", "loan", "cost"
        }

        # Common financial context cue words
        self.financial_context = {
            "loan", "bank", "credit", "debt", "account", "finance", "fund",
            "interest", "payment", "collateral", "investment", "insurance",
            "capital", "rate", "bond", "equity", "mortgage", "repayment", "asset"
        }

    # -------------------------
    # Load main glossary
    # -------------------------
    def _load_base(self, path):
        if path.endswith(".csv"):
            df = pd.read_csv(path)
            df.columns = [c.strip().lower() for c in df.columns]

            term_col = "term" if "term" in df.columns else df.columns[0]
            def_col = "definition" if "definition" in df.columns else df.columns[1]

            for _, row in df.iterrows():
                term = str(row[term_col]).strip()
                definition = str(row[def_col]).strip() if pd.notna(row[def_col]) else ""
                if term:
                    self.glossary[term] = definition

        elif path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                self.glossary.update(json.load(f))
        else:
            raise ValueError("Glossary must be CSV or JSON")

    # -------------------------
    # Extend glossary with FinRAD dataset
    # -------------------------
    def _extend_from_finrad(self, csv_path):
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower() for c in df.columns]

        term_col = next((c for c in df.columns if "term" in c or "word" in c), None)
        def_col = next((c for c in df.columns if "def" in c), None)

        if not term_col:
            print("[WARN] Could not find 'term' column in FinRAD CSV.")
            return

        for _, row in df.iterrows():
            term = str(row[term_col]).strip()
            definition = str(row[def_col]).strip() if def_col and pd.notna(row[def_col]) else ""
            if term and term.lower() not in self.glossary:
                self.glossary[term] = definition

    # -------------------------
    # Check if a term appears in a financial context
    # -------------------------
    def _is_financial_context(self, term, text, window=40):
        matches = [m.start() for m in re.finditer(rf"\b{re.escape(term)}\b", text, re.IGNORECASE)]
        for idx in matches:
            snippet = text[max(0, idx-window): idx+window].lower()
            if any(fin_word in snippet for fin_word in self.financial_context):
                return True
        return False

    # -------------------------
    # Normalize term casing in text
    # -------------------------
    def normalize_terms(self, text):
        for term in self.term_map.keys():
            pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
            text = pattern.sub(term.capitalize(), text)
        return text

    # -------------------------
    # Tag financial terms in text (context-aware)
    # -------------------------
    def tag_terms(self, text):
        found = []
        for term in self.term_map.keys():
            # Find matches
            if re.search(rf"\b{re.escape(term)}\b", text, re.IGNORECASE):
                # Keep multiword terms directly
                if len(term.split()) > 1:
                    found.append(term)
                # For single-word ambiguous terms → only if contextually financial
                elif term.lower() not in self.ambiguous_terms or self._is_financial_context(term, text):
                    found.append(term)
        return found

    # -------------------------
    # Get definition of a term
    # -------------------------
    def get_definition(self, term):
        return self.term_map.get(term.lower(), "")
