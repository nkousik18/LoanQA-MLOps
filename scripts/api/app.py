"""
scripts/api/app.py
────────────────────────────────────────────
Flask backend for the LoanDoc Intelligence Interface.

Supports:
1. Document upload → OCR extraction → vectorstore → LLM
2. LLM-powered Summary / Translation / Explanation
3. Chatbot streaming and UI
4. File reading for full text display
5. Glyph-accurate text map extraction (via PyMuPDF)
"""

import os
import sys
import json
import fitz  # PyMuPDF
import hashlib
import logging
from flask import Flask, jsonify, render_template, request, send_file
from flask_cors import CORS

# ============================================================
#  Ensure proper path imports
# ============================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

# ============================================================
#  Logging Configuration
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================
#  Import Blueprints
# ============================================================
from scripts.api.routes.upload_routes import upload_bp
from scripts.api.routes.llm_routes import llm_bp
from scripts.api.routes.chatbot_routes import chatbot_bp


# ============================================================
#  Flask App Factory
# ============================================================
def create_app():
    """Application factory for the LoanDoc Intelligence Interface."""
    app = Flask(
        __name__,
        static_folder="../../interface/build",
        template_folder="../../interface/build"
    )

    # ✅ Enable CORS
    CORS(app, resources={r"/*": {"origins": "*"}})
    logger.info("🚀 Flask app initialized with CORS enabled.")

    # Register blueprints
    app.register_blueprint(upload_bp, url_prefix="/api")
    app.register_blueprint(llm_bp, url_prefix="/api")
    app.register_blueprint(chatbot_bp, url_prefix="/api")
    logger.info("✅ All route blueprints registered successfully.")

    # ============================================================
    # Health Check Endpoint
    # ============================================================
    @app.route("/api/health", methods=["GET"])
    def health_check():
        logger.info("🩺 Health check requested.")
        return jsonify({"status": "running", "message": "LoanDoc API operational."})

    # ============================================================
    # Unified Pipeline Endpoint
    # ============================================================
    @app.route("/api/process_pdf", methods=["POST"])
    def process_pdf():
        """Upload → Extract → Index"""
        try:
            file = request.files.get("file")
            if not file:
                return jsonify({"error": "No file uploaded"}), 400

            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
            loan_dir = os.path.join(project_root, "data", "loan_docs")
            os.makedirs(loan_dir, exist_ok=True)
            file_path = os.path.join(loan_dir, file.filename)
            file.save(file_path)
            logger.info(f"📥 [Step 1] Saved uploaded file → {file_path}")

            from scripts.extraction_pipeline.main_extractor import process_single_file
            extracted_path = process_single_file(file_path)
            logger.info(f"🧾 Extracted text saved → {extracted_path}")

            from scripts.LLMquery.build_index import add_to_index
            add_to_index(extracted_path)
            logger.info("✅ Vectorstore index updated successfully.")

            return jsonify({
                "status": "✅ Ready for interaction",
                "file": file.filename,
                "message": "Text extracted & indexed. Ready for user interaction."
            }), 200

        except Exception as e:
            logger.exception(" Error in /api/process_pdf")
            return jsonify({"error": str(e)}), 500

    # ============================================================
    # Text-Level LLM Operations
    # ============================================================
    @app.route("/api/process_text", methods=["POST"])
    def process_text():
        try:
            data = request.get_json(force=True)
            text = data.get("text")
            action = data.get("action", "summary").lower()
            language = data.get("language", "English")

            if not text:
                return jsonify({"error": "No text provided"}), 400

            from scripts.LLMquery.prompts.prompt_router import build_prompt
            from scripts.LLMquery.prompts.llm_executor import run_llm

            question_map = {
                "summary": "Summarize this passage concisely, focusing on key financial or ethical aspects.",
                "translate": f"Translate the following passage into {language}.",
                "explain": "Explain the meaning of this passage in simple and clear terms."
            }

            question = question_map.get(action, question_map["summary"])
            prompt_text, intent, conf, gap = build_prompt(
                question=question, docs=[text], mode=action
            )
            response = run_llm(prompt_text)

            return jsonify({
                "action": action,
                "language": language,
                "result": response or prompt_text,
                "confidence": conf,
                "intent": intent
            }), 200

        except Exception as e:
            logger.exception("❌ Error in /api/process_text")
            return jsonify({"error": str(e)}), 500

    # ============================================================
    # Glyph-Level Text Map Extraction (Precise Selection)
    # ============================================================
    @app.route("/api/get_text_map", methods=["POST"])
    def get_text_map():
        """Extracts or serves cached glyph-level text geometry via PyMuPDF."""
        try:
            file = request.files.get("file")
            if not file:
                return jsonify({"error": "No file uploaded"}), 400

            # Compute hash for caching
            file_bytes = file.read()
            pdf_hash = hashlib.sha256(file_bytes).hexdigest()

            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
            cache_dir = os.path.join(project_root, "data", "cache")
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, f"{pdf_hash}.json")

            # Return from cache if exists
            if os.path.exists(cache_path):
                with open(cache_path, "r", encoding="utf-8") as f:
                    return jsonify(json.load(f)), 200

            # Save temporarily
            temp_dir = os.path.join(project_root, "data", "temp")
            os.makedirs(temp_dir, exist_ok=True)
            pdf_path = os.path.join(temp_dir, file.filename)
            with open(pdf_path, "wb") as f:
                f.write(file_bytes)

            logger.info(f"🔍 Extracting glyph geometry for {file.filename}")
            doc = fitz.open(pdf_path)
            pages = []

            for i, page in enumerate(doc):
                text_dict = page.get_text("rawdict")
                page_info = {
                    "page": i + 1,
                    "width": float(page.rect.width),
                    "height": float(page.rect.height),
                    "words": []
                }

                for block in text_dict.get("blocks", []):
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text_value = span.get("text", "").strip()
                            if not text_value:
                                continue
                            x0, y0, x1, y1 = span.get("bbox", [0, 0, 0, 0])
                            page_info["words"].append({
                                "text": text_value,
                                "x": float(x0),
                                "y": float(y0),
                                "width": float(x1 - x0),
                                "height": float(y1 - y0)
                            })

                pages.append(page_info)

            doc.close()
            os.remove(pdf_path)

            result = {"pdf_hash": pdf_hash, "pages": pages}
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(result, f)
            logger.info(f"[CACHE] Saved text map for {file.filename} ({pdf_hash[:12]})")

            return jsonify(result), 200

        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"❌ Error in get_text_map: {e}")
            return jsonify({"error": str(e)}), 500

    # ============================================================
    # Root & Chat Routes
    # ============================================================
    @app.route("/")
    def home():
        logger.info("🏠 Root accessed — frontend runs separately on port 3000.")
        return jsonify({"status": "running", "frontend_url": "http://localhost:3000"})

    @app.route("/chat", methods=["GET"])
    def chat_page():
        logger.info("💬 Serving chat.html interface.")
        return render_template("chat.html")

    return app


# ============================================================
#  App Runner
# ============================================================
if __name__ == "__main__":
    app = create_app()
    logger.info("🚀 LoanDoc Flask API starting on port 8080...")
    app.run(host="0.0.0.0", port=8080, debug=True)
