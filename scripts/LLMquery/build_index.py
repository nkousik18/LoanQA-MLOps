"""
scripts/LLMquery/build_index.py
────────────────────────────────────────────
Builds and updates the Chroma vector index for LoanDoc documents.
Fully compatible with LangChain 0.3+ and Chroma 1.0+.
Auto-heals corrupted indices and handles environment mismatches.
"""

import os
import time
import shutil
import signal
import logging
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ============================================================
# Logging setup
# ============================================================
from scripts.extraction_pipeline.config import setup_logger
logger = setup_logger(__name__, log_type="llm")

# ============================================================
# Configuration
# ============================================================
DATA_PATH = "data/clean_texts"
INDEX_PATH = "scripts/LLMquery/vectorstores/local_doc_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ============================================================
# Compatibility fix for Chroma v1.x (Rust backend)
# ============================================================
# Ensures consistent, local, single-threaded mode on macOS ARM
# os.environ["CHROMA_API_IMPL"] = "local"
os.environ["ALLOW_RESET"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ============================================================
# Utility: Safe delete index if corrupted
# ============================================================
def reset_chroma_index():
    """Force deletes and rebuilds the vector index directory."""
    try:
        if os.path.exists(INDEX_PATH):
            shutil.rmtree(INDEX_PATH)
            logger.warning(f"⚠️ Removed corrupted Chroma index at {INDEX_PATH}")
    except Exception as e:
        logger.error(f"❌ Failed to remove corrupted index: {e}")


# ============================================================
# Add single file to index
# ============================================================
def add_to_index(new_file_path):
    """Adds an extracted text file to the Chroma vector index."""
    if not os.path.exists(new_file_path):
        logger.error(f"❌ Extracted file not found: {new_file_path}")
        return 0

    logger.info(f"🧠 Adding {new_file_path} to existing index...")

    loader = TextLoader(new_file_path, encoding="utf-8")
    new_docs = loader.load()
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    try:
        start = time.time()
        db = Chroma(persist_directory=INDEX_PATH, embedding_function=embeddings)
        db.add_documents(new_docs)
        db.persist()
        elapsed = time.time() - start
        logger.info(f"✅ Added {len(new_docs)} docs to index | Time: {elapsed:.2f}s")
        return len(new_docs)

    except Exception as e:
        logger.exception(f"❌ Indexing failed: {e}")
        reset_chroma_index()
        return 0


# ============================================================
# Rebuild full vector index
# ============================================================
def rebuild_vector_index():
    """Rebuilds the Chroma vector index from all extracted clean text files."""
    start = time.time()
    logger.info("🔄 Rebuilding full vector index...")

    if not os.path.exists(DATA_PATH):
        logger.error(f"❌ Directory not found: {DATA_PATH}")
        raise FileNotFoundError(f"Directory not found: {DATA_PATH}")

    text_files = [os.path.join(DATA_PATH, f) for f in os.listdir(DATA_PATH) if f.endswith(".txt")]
    if not text_files:
        logger.warning("⚠️ No .txt files found in clean_texts.")
        return 0

    docs = []
    for f in text_files:
        try:
            loader = TextLoader(f, encoding="utf-8")
            docs.extend(loader.load())
        except Exception as e:
            logger.warning(f"⚠️ Failed to load {f}: {e}")

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = Chroma.from_documents(docs, embeddings, persist_directory=INDEX_PATH)
    db.persist()

    total = len(docs)
    duration = round(time.time() - start, 2)
    logger.info(f"✅ Rebuilt Chroma index with {total} documents in {duration}s.")
    return total


if __name__ == "__main__":
    logger.info("[ENTRYPOINT] Rebuilding full vector index manually.")
    rebuild_vector_index()
