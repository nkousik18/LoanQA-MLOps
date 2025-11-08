"""
scripts/api/routes/chatbot_routes.py
────────────────────────────────────────────
LLM Query interface for retrieval-augmented QA.
Modernized for LangChain 0.3+ & Chroma 1.3+ (PersistentClient API).
"""

import os, json, time
from flask import Blueprint, request, Response, jsonify, render_template
from scripts.LLMquery.prompts.prompt_router import build_prompt
from scripts.LLMquery.prompts.llm_executor import stream_llm
from scripts.LLMquery.build_index import logger, INDEX_PATH, EMBED_MODEL

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb

chatbot_bp = Blueprint("chatbot", __name__)

# ============================================================
# Chroma + Embeddings Initialization
# ============================================================
def get_chroma_client():
    """Return a persistent local Chroma client (v1.3+)."""
    return chromadb.PersistentClient(path=INDEX_PATH)


embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
client = get_chroma_client()
vectorstore = Chroma(client=client, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

logger.info(f" Loaded persistent Chroma index from: {INDEX_PATH}")

# ============================================================
# Routes
# ============================================================
@chatbot_bp.route("/chat", methods=["GET"])
def open_chat():
    """Serves the interactive chatbot UI."""
    return render_template("chat.html")


@chatbot_bp.route("/query_stream", methods=["POST"])
def query_stream():
    """Streams token-by-token LLM responses to the UI."""
    data = request.get_json()
    question = data.get("question", "").strip()
    mode = data.get("mode", None)

    if not question:
        return jsonify({"error": "Missing question"}), 400

    try:
        start_time = time.time()
        docs = retriever.get_relevant_documents(question)
        prompt, intent, conf, gap = build_prompt(question, docs, mode)

        logger.info(f"[PROMPT USED] Intent='{intent}' | Confidence={conf:.3f}")
        logger.debug(f"Prompt Preview:\n{prompt[:400]}")

        def generate():
            try:
                for token in stream_llm(prompt):
                    yield f"data: {json.dumps({'token': token})}\n\n"

                elapsed = round(time.time() - start_time, 2)
                yield f"data: {json.dumps({'done': True, 'time_taken': elapsed})}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f" Streaming failed: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return Response(generate(), mimetype="text/event-stream")

    except Exception as e:
        logger.error(f" Query failed: {e}")
        return jsonify({"error": str(e)}), 500
