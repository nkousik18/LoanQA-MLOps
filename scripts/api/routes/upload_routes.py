"""
scripts/api/routes/upload_routes.py
────────────────────────────────────────────
Handles file upload → OCR extraction → vector index update.
Now fully compatible with Chroma v1.3+ and LangChain v0.3+.
"""

import os
import time
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

from scripts.extraction_pipeline.main_extractor import process_single_file
from scripts.LLMquery.build_index import logger, INDEX_PATH, EMBED_MODEL
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb

upload_bp = Blueprint("upload", __name__)

# ============================================================
# Constants
# ============================================================
UPLOAD_DIR = "data/loan_docs"
ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg"}


# ============================================================
# Helpers
# ============================================================
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_chroma_client():
    """Return a persistent local Chroma client (v1.3+)."""
    return chromadb.PersistentClient(path=INDEX_PATH)


# ============================================================
# Upload Endpoint
# ============================================================
@upload_bp.route("/upload", methods=["POST"])
def upload_document():
    """Handles upload, OCR extraction, and vector index update."""
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file format"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_DIR, filename)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file.save(save_path)

    logger.info(f"📂 File uploaded: {filename}")

    # ============================================================
    # 1️⃣ Run OCR extraction
    # ============================================================
    start = time.time()
    extracted_path = process_single_file(save_path)
    if not extracted_path:
        return jsonify({"error": "Extraction failed"}), 500

    # ============================================================
    # 2️⃣ Update Vector Index
    # ============================================================
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        client = get_chroma_client()
        db = Chroma(client=client, embedding_function=embeddings)

        # Load new document and add to index
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(extracted_path, encoding="utf-8")
        new_docs = loader.load()
        db.add_documents(new_docs)

        elapsed = round(time.time() - start, 2)
        logger.info(f"✅ Indexed {len(new_docs)} docs in {elapsed}s.")
        return jsonify({
            "message": f"File processed successfully in {elapsed}s",
            "content_path": extracted_path
        }), 200

    except Exception as e:
        logger.exception(f"❌ Indexing failed: {e}")
        return jsonify({"error": str(e)}), 500
