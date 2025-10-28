import os
import time
import shutil
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ============================================================
# Logging Setup
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
# Add to Index (Self-Healing)
# ============================================================
def add_to_index(new_file_path):
    """
    Adds a single extracted text file to the existing Chroma index.
    Automatically rebuilds if corruption (TypeError len()) is detected.
    """
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
        logger.info(f"✅ Added {len(new_docs)} docs to index from {new_file_path} | Time: {elapsed:.2f}s")

    except TypeError as e:
        # Known Chroma SQLite bug — invalid sequence ID decoding
        if "len()" in str(e):
            logger.warning("⚠️ Detected Chroma corruption — rebuilding vectorstore from scratch...")
            shutil.rmtree(INDEX_PATH, ignore_errors=True)

            db = Chroma.from_documents(new_docs, embeddings, persist_directory=INDEX_PATH)
            db.persist()
            logger.info(f"✅ Rebuilt fresh index with {new_file_path}")
        else:
            logger.exception(f"❌ Unexpected TypeError while adding {new_file_path}: {e}")
            raise e

    except Exception as e:
        logger.exception(f"❌ Failed to index {new_file_path}: {e}")
        raise e

    return len(new_docs)


# ============================================================
# Rebuild Entire Index
# ============================================================
def rebuild_vector_index():
    """Rebuilds the Chroma vector index from extracted clean text files."""
    start = time.time()
    logger.info("🔄 Rebuilding vector index from extracted documents...")

    if not os.path.exists(DATA_PATH):
        logger.error(f"❌ Directory not found: {DATA_PATH}")
        raise FileNotFoundError(f"Directory not found: {DATA_PATH}")

    text_files = [
        os.path.join(DATA_PATH, f)
        for f in os.listdir(DATA_PATH)
        if f.endswith(".txt")
    ]
    if not text_files:
        logger.warning("⚠️ No .txt files found in clean_texts.")
        return 0

    docs = []
    for f in text_files:
        try:
            loader = TextLoader(f, encoding="utf-8")
            docs.extend(loader.load())
            logger.debug(f"✅ Loaded: {f}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load {f}: {e}", exc_info=True)

    logger.info(f"🧾 Total documents loaded: {len(docs)}")

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = Chroma.from_documents(docs, embeddings, persist_directory=INDEX_PATH)
    db.persist()

    total = len(docs)
    duration = round(time.time() - start, 2)

    # Log embedding dimension for debugging
    try:
        sample_vec = embeddings.embed_query("Test")
        logger.info(f"📏 Embedding Dimension: {len(sample_vec)} | Total Docs: {total}")
    except Exception:
        logger.debug("Embedding dimension check skipped.")

    logger.info(f"✅ Rebuilt index with {total} documents in {duration}s.")
    return total


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    logger.info("[ENTRYPOINT] Rebuilding full vector index for LLMquery module")
    rebuild_vector_index()
