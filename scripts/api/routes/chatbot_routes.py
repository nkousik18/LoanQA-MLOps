from flask import Blueprint, request, Response, jsonify, render_template
import json, time
from scripts.LLMquery.prompts.prompt_router import build_prompt
from scripts.LLMquery.prompts.math_utils import evaluate_math
from scripts.LLMquery.build_index import logger
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from scripts.LLMquery.prompts.llm_executor import stream_llm  # ✅ new streaming version
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


chatbot_bp = Blueprint("chatbot", __name__)

# Vectorstore setup
CHROMA_PATH = "scripts/LLMquery/vectorstores/local_doc_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
vectorstore = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


@chatbot_bp.route("/chat", methods=["GET"])
def open_chat():
    return render_template("chat.html")


@chatbot_bp.route("/query_stream", methods=["POST"])
def query_stream():
    """Streams the LLM response token-by-token to the frontend."""
    data = request.get_json()
    question = data.get("question", "").strip()
    mode = data.get("mode", None)

    if not question:
        return jsonify({"error": "Missing question"}), 400

    try:
        start_time = time.time()
        docs = retriever.get_relevant_documents(question)
        prompt, intent, conf, gap = build_prompt(question, docs, mode)
        logger.info(f"[PROMPT USED] Intent='{intent}' | Prompt Preview:\n{prompt[:500]}")
        logger.info(f"🧭 Intent={intent} | Confidence={conf:.3f} | Gap={gap:.3f}")

        def generate():
            """Yield streaming tokens to the client."""
            try:
                for token in stream_llm(prompt):
                    yield f"data: {json.dumps({'token': token})}\n\n"

                elapsed = round(time.time() - start_time, 2)
                yield f"data: {json.dumps({'done': True, 'time_taken': elapsed})}\n\n"
                yield "data: [DONE]\n\n"

            except Exception as e:
                logger.error(f"❌ Streaming failed: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return Response(generate(), mimetype="text/event-stream")

    except Exception as e:
        logger.error(f"❌ Query failed: {e}")
        return jsonify({"error": str(e)}), 500
