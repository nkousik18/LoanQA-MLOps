from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import os
from scripts.extraction_pipeline.main_extractor import process_single_file
from scripts.LLMquery.build_index import add_to_index, logger

upload_bp = Blueprint("upload", __name__)
UPLOAD_FOLDER = "data/loan_docs/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@upload_bp.route("/upload", methods=["POST"])
def upload_file():
    """Handles PDF or image uploads, runs OCR, and returns the extracted text path."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    try:
        # Run OCR → get extracted .txt path
        extracted_file_path = process_single_file(file_path)
        logger.info(f"[DEBUG] Extracted file path returned: {extracted_file_path}")

        # Add to vector index
        add_to_index(extracted_file_path)
        logger.info(f"✅ Indexed document successfully: {filename}")

        return jsonify({
            "status": "success",
            "filename": filename,
            "message": "Document processed successfully.",
            "content_path": extracted_file_path  # ✅ Return path here
        }), 200

    except Exception as e:
        logger.error(f"[UPLOAD ERROR] {e}")
        return jsonify({"error": str(e)}), 500
