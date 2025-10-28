# scripts/LLMquery/embeddings_index.py

import os
import time
from scripts.extraction_pipeline.config import setup_logger
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ============================================================
# Initialize Logger (LLM Log Group)
# ============================================================
logger = setup_logger(__name__, log_type="llm")

# ============================================================
# Configuration
# ============================================================
DEFAULT_INDEX_PATH = "scripts/LLMquery/vectorstores/local_doc_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ============================================================
# Load Vectorstore
# ============================================================
def load_vectorstore(index_path: str = DEFAULT_INDEX_PATH):
    """
    Loads a Chroma vectorstore and its embeddings.
    Used by the Loan Document Assistant for retrieval-based QA.
    """
    start_time = time.time()
    logger.info(f"📂 Attempting to load Chroma vectorstore from: {index_path}")

    if not os.path.exists(index_path):
        logger.error(f"❌ Vectorstore path not found: {index_path}")
        raise FileNotFoundError(f"Vectorstore path not found: {index_path}")

    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        db = Chroma(persist_directory=index_path, embedding_function=embeddings)

        count = db._collection.count() if hasattr(db, "_collection") else "unknown"
        elapsed = round(time.time() - start_time, 2)

        logger.info(f"✅ Loaded Chroma DB successfully | Entries: {count} | Model: {EMBED_MODEL}")
        logger.info(f"⏱️ Load time: {elapsed}s")

        return db, embeddings

    except Exception as e:
        logger.exception(f"❌ Failed to load Chroma vectorstore at {index_path}: {e}")
        raise e


# ============================================================
# CLI Entry Point (for manual validation)
# ============================================================
if __name__ == "__main__":
    logger.info("[ENTRYPOINT] Loading vectorstore manually for validation...")
    try:
        db, emb = load_vectorstore()
        logger.info("🎯 Vectorstore loaded successfully via CLI test.")
    except Exception as e:
        logger.error(f"Vectorstore load failed: {e}", exc_info=True)
