"""
Hybrid chunking strategy for LoanDocQA+
 - Sentence-aware
 - Page-aware
 - Fixed-size fallback
"""

import re
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt', quiet=True)

# ------------------------------------------------------------
# Clean text (remove nulls, multiple spaces, PDF artifacts)
# ------------------------------------------------------------
def clean_text(t: str) -> str:
    t = t.replace("\x00", "")
    t = re.sub(r"[ ]{2,}", " ", t)
    t = re.sub(r"[^\S\r\n]+", " ", t)
    return t.strip()


# ------------------------------------------------------------
# Page-aware split (based on PAGE markers or large gaps)
# ------------------------------------------------------------
def split_by_pages(text: str):
    if "=== page" in text.lower():
        return re.split(r"=+\s*page.*=+", text, flags=re.I)
    return [text]


# ------------------------------------------------------------
# Sentence-aware chunking
# ------------------------------------------------------------
def sentence_chunks(text: str, max_tokens=180):
    sents = sent_tokenize(text)
    chunks, buf = [], ""

    for s in sents:
        if len(buf.split()) + len(s.split()) > max_tokens:
            chunks.append(buf.strip())
            buf = s
        else:
            buf += " " + s
    if buf.strip():
        chunks.append(buf.strip())

    return chunks


# ------------------------------------------------------------
# Fixed-size fallback
# ------------------------------------------------------------
def fixed_chunks(text: str, size=180):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]


# ------------------------------------------------------------
# Hybrid chunker
# ------------------------------------------------------------
def smart_chunk(raw: str):
    raw = clean_text(raw)

    pages = split_by_pages(raw)

    final_chunks = []

    for pg in pages:
        pg = pg.strip()
        if not pg:
            continue

        # sentence-aware first
        sent_chunks = sentence_chunks(pg)
        if len(sent_chunks) > 0:
            final_chunks.extend(sent_chunks)
        else:
            final_chunks.extend(fixed_chunks(pg))

    # remove extremely small noise chunks
    final_chunks = [c for c in final_chunks if len(c.split()) > 5]

    return final_chunks
