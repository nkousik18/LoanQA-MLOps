"""
llm_routes.py
────────────────────────────────────────────
Endpoints for Summary, Translation, and Explanation features.
Each endpoint uses its fine-tuned, purpose-specific prompt from
the scripts/LLMquery/prompts/ directory and executes it via run_llm().
"""

from flask import Blueprint, request, jsonify
import logging

# Core LLM executor
from scripts.LLMquery.prompts.llm_executor import run_llm

# Fine-tuned prompt templates
from scripts.LLMquery.prompts.summary_basic import summary_prompt
from scripts.LLMquery.prompts.translation_basic import translation_prompt
from scripts.LLMquery.prompts.explanation_basic import explanation_prompt

# Initialize logger
logger = logging.getLogger(__name__)

# Flask Blueprint registration
llm_bp = Blueprint("llm", __name__)


# ============================================================
# 1️⃣  SUMMARY ROUTE
# ============================================================
@llm_bp.route("/summary", methods=["POST"])
def summarize_text():
    """Generate a concise summary for the selected text."""
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "Missing text"}), 400

    try:
        # Build simplified summary prompt
        prompt = summary_prompt(text)
        logger.info(f"[SUMMARY] Built prompt for text of length {len(text)}")

        # Execute LLM query
        response = run_llm(prompt)
        logger.info("[SUMMARY] Response generated successfully.")

        return jsonify({"summary": response}), 200

    except Exception as e:
        logger.error(f"[SUMMARY ERROR] {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================
# 2️⃣  TRANSLATION ROUTE
# ============================================================
@llm_bp.route("/translate", methods=["POST"])
def translate_text():
    """Translate selected text into the requested target language."""
    data = request.get_json()
    text = data.get("text", "")
    lang = data.get("lang", "en")

    if not text:
        return jsonify({"error": "Missing text"}), 400

    try:
        # Build simplified translation prompt
        prompt = translation_prompt(text, lang)
        logger.info(f"[TRANSLATE] Building translation prompt for lang={lang}")

        # Execute LLM query
        response = run_llm(prompt)
        logger.info("[TRANSLATE] Translation completed successfully.")

        return jsonify({"translation": response}), 200

    except Exception as e:
        logger.error(f"[TRANSLATE ERROR] {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================
# 3️⃣  EXPLANATION ROUTE
# ============================================================
@llm_bp.route("/explain", methods=["POST"])
def explain_text():
    """Explain complex clauses or terminology in simpler terms."""
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "Missing text"}), 400

    try:
        # Build simplified explanation prompt
        prompt = explanation_prompt(text)
        logger.info(f"[EXPLAIN] Built explanation prompt for text length {len(text)}")

        # Execute LLM query
        response = run_llm(prompt)
        logger.info("[EXPLAIN] Explanation generated successfully.")

        return jsonify({"explanation": response}), 200

    except Exception as e:
        logger.error(f"[EXPLAIN ERROR] {e}")
        return jsonify({"error": str(e)}), 500
