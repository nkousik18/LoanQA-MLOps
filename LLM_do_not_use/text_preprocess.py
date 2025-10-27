# LLMquery/text_preprocess.py

import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

def clean_text(raw_text: str) -> str:
    """
    Cleans raw OCR text by removing extra spaces, symbols, and non-ASCII chars.
    """
    text = re.sub(r'\s+', ' ', raw_text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100):
    """
    Splits text into overlapping chunks to prepare for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", ".", "!", "?"]
    )
    chunks = splitter.split_text(text)
    return chunks
