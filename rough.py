import os
import pandas as pd
import re

class FinancialGlossary:
    def __init__(self, glossary_path=None, finrad_csv=None):
        """
        glossary_path: path to your existing glossary (CSV or JSON)
        finrad_csv: optional path to FinRAD dataset file
        """
        self.glossary = {}
        if glossary_path:
            self._load_base(glossary_path)
        if finrad_csv:
            self._extend_from_finrad(finrad_csv)

        # lowercase mapping for matching
        self.term_map = {term.lower(): definition for term, definition in self.glossary.items()}

    def _load_base(self, path):
        if path.endswith(".csv"):
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                self.glossary[row["terms"]] = row["definitions"]
        elif path.endswith(".json"):
            import json
            with open(path, "r", encoding="utf-8") as f:
                self.glossary.update(json.load(f))
        else:
            raise ValueError("Glossary must be CSV or JSON")

    def _extend_from_finrad(self, csv_path):
        """
        Read the FinRAD dataset and extract terms to add to glossary.
        FinRAD likely has a column 'words' or similar. We'll treat them as terms needing definitions.
        """
        df = pd.read_csv(csv_path)
        # Look at columns to find plausible term column names
        # Suppose there's a column 'word' in dataset
        if "word" in df.columns:
            words = df["word"].unique()
            for w in words:
                w_clean = str(w).strip()
                if w_clean and w_clean.lower() not in self.term_map:
                    self.glossary[w_clean] = ""  # placeholder; you can fill definitions later
        else:
            print("[WARN] 'word' column not found in FinRAD; check dataset schema")

    def normalize_terms(self, text):
        for term in self.term_map.keys():
            pattern = re.compile(rf"\b{term}\b", re.IGNORECASE)
            # only normalize if definition exists or term is known
            text = pattern.sub(term.capitalize(), text)
        return text

    def tag_terms(self, text):
        found = []
        for term in self.term_map.keys():
            if re.search(rf"\b{term}\b", text, re.IGNORECASE):
                found.append(term)
        return found

    def get_definition(self, term):
        return self.term_map.get(term.lower())