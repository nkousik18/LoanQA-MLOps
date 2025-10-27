import os, time, json, csv, asyncio
from datetime import datetime
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

# LangChain and LLM imports
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler

# Internal imports
from LLMquery.prompts.prompt_router import build_prompt
from LLMquery.prompts.math_utils import evaluate_math
from extraction_pipeline.main_extractor import main as run_extraction
from LLMquery.build_index import rebuild_vector_index

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
# 2️⃣ Configuration
# ============================================================
CHROMA_PATH = "LLMquery/vectorstores/loan_doc_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "phi3"
MAX_TOKENS = 512
LOG_PATH = "logs/query_logs.csv"
UPLOAD_DIR = "data/loan_docs"

os.makedirs("logs", exist_ok=True)

# ============================================================
# 3️⃣ Load Vectorstore and Model
# ============================================================
print("📂 Initializing vectorstore and embeddings...")
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print(f"✅ Chroma entries: {vectorstore._collection.count()}")

print(f"🧠 Loading LLM model: {LLM_MODEL}")
llm = OllamaLLM(model=LLM_MODEL, temperature=0.2, num_predict=MAX_TOKENS)
print("✅ Model ready.\n")

# ============================================================
# 4️⃣ Utilities
# ============================================================
def log_to_csv(entry: dict):
    """Append query metadata to CSV."""
    header = ["timestamp", "question", "intent", "confidence", "gap", "mode",
              "response_length", "time_taken_sec", "sources", "prompt", "answer"]
    file_exists = os.path.isfile(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(entry)


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
# 5️⃣ Upload → Extract → Rebuild
# ============================================================
@app.post("/upload_file")
@app.post("/upload_file")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload new loan document → extract → embed → add to existing index.
    """
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Save uploaded file
    with open(file_path, "wb") as f:
        f.write(await file.read())
    print(f"📄 Uploaded file saved to: {file_path}")

    try:
        # Step 1: Extract only this file
        from extraction_pipeline.main_extractor import process_single_file
        print("🔍 Extracting text from uploaded file...")
        extracted_path = process_single_file(file_path)
        if not extracted_path:
            return JSONResponse({"status": "error", "message": "Extraction failed."}, status_code=500)

        # Step 2: Add extracted file to index

        from LLMquery.build_index import add_to_index
        print("🧠 Adding new file to vector index...")
        added = add_to_index(extracted_path)

        return {
            "status": "success",
            "uploaded_file": file.filename,
            "indexed_docs": added,
            "message": "✅ File uploaded, extracted, and indexed successfully."
        }

    except Exception as e:
        print(f"⚠️ Error during upload pipeline: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)



# ============================================================
# 6️⃣ Query Endpoint — Async Streaming
# ============================================================
from functools import lru_cache
@lru_cache(maxsize=32)
def cached_retrieval(query: str):
    return retriever.get_relevant_documents(query)


@app.post("/query_stream")
async def query_stream(request: Request):
    """Asynchronously streams answers with token updates."""
    data = await request.json()
    question = data.get("question", "").strip()
    mode = data.get("mode", None)
    if not question:
        return {"answer": "Please provide a valid question."}

    start_time = time.time()
    print(f"🧠 Query: {question}")

    docs = cached_retrieval(question)
    print(f"📄 Retrieved {len(docs)} documents")

    prompt, intent, conf, gap = build_prompt(question, docs, mode)
    print(f"🧭 Intent={intent} | Confidence={conf}")

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

            # Optional: Math verification for finance
            if intent == "finance":
                try:
                    verified = evaluate_math(response)
                    if verified:
                        response += "\n\n💡 **Verified Computation:**\n" + "\n".join(f"- {r}" for r in verified)
                except Exception as e:
                    print(f"⚠️ Math verification error: {e}")

            # Sources + timing + log
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
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(token_stream(), media_type="text/event-stream")
