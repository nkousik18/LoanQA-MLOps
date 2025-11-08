"""
scripts/LLMquery/build_index.py

Builds and updates the Chroma vector index for LoanDoc documents.

"""

import os
import time
import shutil
import logging
import chromadb
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from scripts.extraction_pipeline.config import setup_logger

# ============================================================
# Logging setup
# ============================================================
logger = setup_logger(__name__, log_type="llm")

# ============================================================
# Configuration
# ============================================================
DATA_PATH = "data/clean_texts"
INDEX_PATH = "scripts/LLMquery/vectorstores/local_doc_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ============================================================
# Utility
# ============================================================
def reset_chroma_index():
    """Deletes the existing index if corrupted."""
    if os.path.exists(INDEX_PATH):
        shutil.rmtree(INDEX_PATH)
        logger.warning(f"⚠️ Removed corrupted Chroma index at {INDEX_PATH}")

# ============================================================
# New Client Constructor (Chroma 1.3+)
# ============================================================
def get_chroma_client():
    """Creates a modern persistent Chroma client (v1.3+ API)."""
    import chromadb
    return chromadb.PersistentClient(path=INDEX_PATH)


# ============================================================
# Add a single file to the index
# ============================================================
def add_to_index(new_file_path):
    if not os.path.exists(new_file_path):
        logger.error(f"❌ File not found: {new_file_path}")
        return 0

    logger.info(f"🧠 Adding {new_file_path} to index...")
    loader = TextLoader(new_file_path, encoding="utf-8")
    new_docs = loader.load()
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    try:
        start = time.time()
        client = get_chroma_client()
        db = Chroma(client=client, embedding_function=embeddings)
        db.add_documents(new_docs)
        elapsed = time.time() - start
        logger.info(f"✅ Added {len(new_docs)} docs to index | Time: {elapsed:.2f}s")
        return len(new_docs)
    except Exception as e:
        logger.exception(f"❌ Indexing failed: {e}")
        reset_chroma_index()
        return 0

# ============================================================
# Rebuild the entire index
# ============================================================
def rebuild_vector_index():
    logger.info("🔄 Rebuilding full Chroma vector index...")
    start = time.time()

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"❌ Directory not found: {DATA_PATH}")

    text_files = [
        os.path.join(DATA_PATH, f)
        for f in os.listdir(DATA_PATH)
        if f.endswith(".txt")
    ]
    if not text_files:
        logger.warning("⚠️ No .txt files found for indexing.")
        return 0

    docs = []
    for file_path in text_files:
        try:
            loader = TextLoader(file_path, encoding="utf-8")
            docs.extend(loader.load())
        except Exception as e:
            logger.warning(f"⚠️ Failed to load {file_path}: {e}")

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    client = get_chroma_client()
    db = Chroma.from_documents(docs, embeddings, client=client)   # ✅ fixed line
    total = len(docs)
    duration = round(time.time() - start, 2)
    logger.info(f"✅ Rebuilt Chroma index with {total} docs in {duration}s.")
    return total


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    logger.info("[ENTRYPOINT] Manual vector index rebuild requested.")
    rebuild_vector_index()
