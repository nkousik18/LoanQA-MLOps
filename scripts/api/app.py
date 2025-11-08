"""
scripts/api/app.py

Flask backend for the LoanDoc Intelligence Interface.

Serves endpoints for:
1. Document upload and OCR extraction
2. LLM-powered Summary / Translation / Explanation
3. Chatbot streaming and UI
4. File reading for full text display
"""

import os
import logging
from flask import Flask, jsonify, render_template, request, send_file
from flask_cors import CORS

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# Import Blueprints
from scripts.api.routes.upload_routes import upload_bp
from scripts.api.routes.llm_routes import llm_bp
from scripts.api.routes.chatbot_routes import chatbot_bp


# Flask App Factory
def create_app():
    """Application factory for the LoanDoc Intelligence Interface."""
    app = Flask(
        __name__,
        template_folder="../../interface/templates",
        static_folder="../../interface/static"
    )

    # Enable CORS for frontend access
    CORS(app)
    logger.info("Flask app initialized with CORS enabled.")

    # Register blueprints
    app.register_blueprint(upload_bp, url_prefix="/api")
    app.register_blueprint(llm_bp, url_prefix="/api")
    app.register_blueprint(chatbot_bp, url_prefix="/api")
    logger.info("All route blueprints registered successfully.")

    # Health Check Endpoint
    @app.route("/api/health", methods=["GET"])
    def health_check():
        """Simple health check endpoint."""
        logger.info("Health check requested.")
        return jsonify({"status": "running", "message": "LoanDoc API operational."})

    # Read Extracted Text Endpoint (used by upload_text_viewer.js)
    @app.route("/api/read_text", methods=["GET"])
    def read_text():
        """Safely streams the full extracted text file back to the UI."""
        path = request.args.get("path")
        if not path:
            logger.error("Missing file path in /api/read_text request.")
            return "Missing file path", 400

        # Detect the absolute project root robustly
        # Start from this file's location and go up until 'data' folder is found
        current_dir = os.path.abspath(os.path.dirname(__file__))
        while current_dir and not os.path.isdir(os.path.join(current_dir, "data")):
            parent = os.path.dirname(current_dir)
            if parent == current_dir:
                break  # Stop at system root
            current_dir = parent

        project_root = current_dir
        clean_dir = os.path.join(project_root, "data", "clean_texts")
        abs_path = os.path.abspath(path)

        logger.debug(f"[DEBUG] project_root = {project_root}")
        logger.debug(f"[DEBUG] clean_dir = {clean_dir}")
        logger.debug(f"[DEBUG] abs_path  = {abs_path}")

        # Security check: ensure the path is within clean_texts
        if not abs_path.startswith(os.path.abspath(clean_dir)):
            logger.warning(f"Attempted access outside clean_texts: {abs_path}")
            return "Access denied", 403

        # Verify file exists
        if not os.path.exists(abs_path):
            logger.error(f"File not found: {abs_path}")
            return "File not found", 404

        logger.info(f"Streaming extracted text file: {abs_path}")
        return send_file(abs_path, mimetype="text/plain")

    # UI Routes
    @app.route("/")
    def home():
        """Serves the main LoanDoc Intelligence interface."""
        logger.info("Serving index.html for main interface.")
        return render_template("index.html")

    @app.route("/api/chat", methods=["GET"])
    def chat_page():
        """Serves the chatbot interface (chat.html)."""
        logger.info("Serving chat.html interface.")
        return render_template("chat.html")

    return app


# App Runner
if __name__ == "__main__":
    app = create_app()
    logger.info("LoanDoc Flask API starting on port 8000...")
    app.run(host="0.0.0.0", port=8000, debug=True)