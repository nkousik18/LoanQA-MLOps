import os, time, json, csv, asyncio
from datetime import datetime
from functools import lru_cache
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

# LangChain / LLM imports
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler

# Internal imports
from scripts.LLMquery.prompts.prompt_router import build_prompt
from scripts.LLMquery.prompts.math_utils import evaluate_math
from scripts.extraction_pipeline.config import setup_logger

# ============================================================
# 1️⃣ FastAPI Initialization
# ============================================================
app = FastAPI(title="LoanDocQA+ Unified Pipeline", version="7.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 2️⃣ Configuration & Logging
# ============================================================
logger = setup_logger(__name__, log_type="llm")

CHROMA_PATH = "scripts/LLMquery/vectorstores/local_doc_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "phi3"
MAX_TOKENS = 512
LOG_PATH = "logs/query_logs.csv"
UPLOAD_DIR = "data/loan_docs"

os.makedirs("logs", exist_ok=True)

logger.info("🚀 LoanDocQA+ API initialized.")
logger.info(f"CHROMA_PATH={CHROMA_PATH}, LLM_MODEL={LLM_MODEL}, EMBED_MODEL={EMBED_MODEL}")

# ============================================================
# 3️⃣ Load Vectorstore and Model
# ============================================================
try:
    logger.info("📂 Loading vectorstore and embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    logger.info(f"✅ Vectorstore ready with {vectorstore._collection.count()} entries.")
except Exception as e:
    logger.exception(f"❌ Failed to initialize vectorstore: {e}")
    raise e

try:
    logger.info(f"🧠 Loading LLM model: {LLM_MODEL}")
    llm = OllamaLLM(model=LLM_MODEL, temperature=0.2, num_predict=MAX_TOKENS)
    logger.info("✅ LLM initialized successfully.")
except Exception as e:
    logger.exception(f"❌ Failed to initialize LLM: {e}")
    raise e

# ============================================================
# 4️⃣ Utility Functions
# ============================================================
def log_to_csv(entry: dict):
    """Append query metadata to CSV."""
    header = [
        "timestamp", "question", "intent", "confidence", "gap", "mode",
        "response_length", "time_taken_sec", "sources", "prompt", "answer"
    ]
    file_exists = os.path.isfile(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(entry)
    logger.debug(f"🪶 Logged query metadata for '{entry.get('question', '')[:40]}...'")

def format_sources(docs):
    """Generate structured sources for UI."""
    sources_md, structured = [], []
    for doc in docs[:3]:
        snippet = doc.page_content.strip().replace("\n", " ")
        snippet = snippet[:400] + "..." if len(snippet) > 400 else snippet
        fname = doc.metadata.get("source") or "unknown_source"
        sources_md.append(f"**{fname}** — {snippet}")
        structured.append({"file": fname, "snippet": snippet})
    return "\n\n".join(sources_md), structured


# ============================================================
# 5️⃣ Upload → Extract → Rebuild Index
# ============================================================
@app.post("/upload_file")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload new loan document → extract → embed → add to existing index.
    """
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    logger.info(f"📄 Upload received: {file.filename}")

    try:
        # Save uploaded file
        with open(file_path, "wb") as f:
            f.write(await file.read())
        logger.info(f"✅ File saved to: {file_path}")

        # Step 1: Extract text
        from scripts.extraction_pipeline.main_extractor import process_single_file
        logger.info("🔍 Extracting text from uploaded file...")
        extracted_path = process_single_file(file_path)
        if not extracted_path:
            logger.error("❌ Extraction returned empty result.")
            return JSONResponse({"status": "error", "message": "Extraction failed."}, status_code=500)

        # Step 2: Add to vector index
        from scripts.LLMquery.build_index import add_to_index
        logger.info("🧠 Adding new file to Chroma vector index...")
        added = add_to_index(extracted_path)
        logger.info(f"✅ Indexed {added} new document(s).")

        return {
            "status": "success",
            "uploaded_file": file.filename,
            "indexed_docs": added,
            "message": "✅ File uploaded, extracted, and indexed successfully."
        }

    except Exception as e:
        logger.exception(f"⚠️ Error during upload pipeline for {file.filename}: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# ============================================================
# 6️⃣ Query Endpoint — Async Streaming
# ============================================================
@lru_cache(maxsize=32)
def cached_retrieval(query: str):
    logger.debug(f"🔍 Cached retrieval called for: '{query[:60]}...'")
    return retriever.get_relevant_documents(query)


@app.post("/query_stream")
async def query_stream(request: Request):
    """Asynchronously streams answers with token updates."""
    data = await request.json()
    question = data.get("question", "").strip()
    mode = data.get("mode", None)

    if not question:
        logger.warning("⚠️ Empty query received.")
        return {"answer": "Please provide a valid question."}

    start_time = time.time()
    logger.info(f"🧠 New query: '{question}' | Mode: {mode or 'auto'}")

    try:
        docs = cached_retrieval(question)
        logger.info(f"📄 Retrieved {len(docs)} document(s) for context.")

        prompt, intent, conf, gap = build_prompt(question, docs, mode)
        logger.info(f"🧭 Intent={intent} | Confidence={conf:.3f} | Gap={gap:.3f}")

    except Exception as e:
        logger.exception(f"❌ Retrieval or prompt building failed: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

    async def token_stream():
        callback = AsyncIteratorCallbackHandler()
        gen_task = asyncio.create_task(llm.agenerate([prompt], callbacks=[callback]))
        response_text = ""

        try:
            async for chunk in callback.aiter():
                response_text += chunk
                yield f"data: {json.dumps({'token': chunk})}\n\n"

            await gen_task
            response = response_text.strip()

            # Optional: Math verification for finance intent
            if intent == "finance":
                try:
                    verified = evaluate_math(response)
                    if verified:
                        response += "\n\n💡 **Verified Computation:**\n" + "\n".join(f"- {r}" for r in verified)
                        logger.debug(f"🔢 Math verification succeeded for finance intent.")
                except Exception as e:
                    logger.warning(f"⚠️ Math verification failed: {e}")

            # Source summary + CSV log
            sources_md, sources_struct = format_sources(docs)
            elapsed = round(time.time() - start_time, 2)

            log_to_csv({
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "intent": intent,
                "confidence": conf,
                "gap": gap,
                "mode": mode or "auto",
                "response_length": len(response),
                "time_taken_sec": elapsed,
                "sources": sources_md.replace("\n", " "),
                "prompt": prompt[:4000],
                "answer": response[:4000],
            })

            logger.info(
                f"✅ Query completed | Intent={intent} | Duration={elapsed}s | "
                f"ResponseLen={len(response)} | Confidence={conf:.3f}"
            )

            final_payload = {
                "answer": response,
                "sources": sources_struct,
                "intent": intent,
                "confidence": conf,
                "gap": gap,
                "time_taken_sec": elapsed,
            }

            yield f"data: {json.dumps(final_payload)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.exception(f"❌ Streaming or response generation failed: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(token_stream(), media_type="text/event-stream")
