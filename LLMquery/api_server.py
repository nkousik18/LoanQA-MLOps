import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from LLMquery.prompts.build_prompt import build_prompt  # ✅ new central router
from LLMquery.prompts.math_utils import evaluate_math

# ============================================================
# 1️⃣ Initialize FastAPI app
# ============================================================
app = FastAPI(title="LoanDocQA+ API (Multi-Mode Prompting + SymPy Evaluator)", version="5.0")

# ✅ Allow cross-origin access
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
LLM_MODEL = "phi3"   # 🔁 general reasoning model
MAX_TOKENS = 512
MEMORY_LIMIT = 5

# ============================================================
# 3️⃣ Load embeddings, vectorstore, and retriever
# ============================================================
print("📂 Loading Chroma vectorstore and Embeddings...")
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
print(f"✅ Loaded Chroma DB with {vectorstore._collection.count()} entries.\n")

# ============================================================
# 4️⃣ Initialize LLM (direct reasoning mode)
# ============================================================
print(f"🧠 Loading LLM model: {LLM_MODEL}")
llm = OllamaLLM(model=LLM_MODEL, temperature=0.2, num_predict=MAX_TOKENS)
print("✅ LLM loaded successfully.\n")
print("⚙️ Running in direct reasoning mode (multi-prompt architecture).\n")

# ============================================================
# 5️⃣ In-memory chat history
# ============================================================
conversation_history = []  # [{"role": "user"/"assistant", "content": "..."}]

def add_to_history(role: str, content: str):
    """Store conversation turns and keep memory limited."""
    conversation_history.append({"role": role, "content": content})
    if len(conversation_history) > MEMORY_LIMIT * 2:
        del conversation_history[:2]

def get_history_text():
    """Return formatted conversation memory for context."""
    if not conversation_history:
        return ""
    return "\n\n".join(
        f"{msg['role'].upper()}: {msg['content']}" for msg in conversation_history
    )

# ============================================================
# 6️⃣ Root endpoint
# ============================================================
@app.get("/")
def home():
    return {
        "message": "✅ LoanDocQA+ API (Multi-Prompt + SymPy Evaluator) is running.",
        "vectorstore_size": vectorstore._collection.count(),
        "model": LLM_MODEL,
    }

# ============================================================
# 7️⃣ Query endpoint — multi-prompt, math-verified
# ============================================================
@app.post("/query_stream")
async def query_stream(request: Request):
    """Answer user queries using smart prompt routing + math verification."""
    data = await request.json()
    question = data.get("question", "").strip()
    mode = data.get("mode", None)  # ✅ allow explicit mode override
    if not question:
        return {"answer": "Please provide a valid question."}

    start_time = time.time()
    print(f"🧠 Query received: {question}")
    if mode:
        print(f"🎯 Mode override: {mode}")

    # Step 1️⃣ Retrieve relevant docs
    docs = retriever.get_relevant_documents(question)
    print(f"📄 Retrieved {len(docs)} docs")

    # Step 2️⃣ Build prompt dynamically (auto or manual mode)
    prompt = build_prompt(question, docs, mode)
    print("🧩 Prompt built successfully using adaptive logic.")

    # Step 3️⃣ Generate response from LLM
    try:
        response = llm.invoke(prompt)
    except Exception as e:
        print(f"⚠️ LLM execution failed: {e}")
        response = "An error occurred while processing your query."

    # Step 4️⃣ Evaluate math expressions for verification (finance mode)
    computed_results = evaluate_math(response)
    if computed_results:
        response += "\n\n💡 **Verified Computation:**\n" + "\n".join(
            f"- {r}" for r in computed_results
        )

    # Step 5️⃣ Update memory
    add_to_history("user", question)
    add_to_history("assistant", response)

    # Step 6️⃣ Prepare source snippets for UI
    sources = []
    for doc in docs[:3]:
        snippet = doc.page_content.strip().replace("\n", " ")
        snippet = snippet[:400] + "..." if len(snippet) > 400 else snippet
        file_name = (
            doc.metadata.get("source")
            or doc.metadata.get("file")
            or "unknown_source"
        )
        sources.append({"file": file_name, "snippet": snippet})

    # Step 7️⃣ Timing + Return
    time_taken = round(time.time() - start_time, 2)
    print(f"✅ Completed in {time_taken}s | Answer length: {len(response)}")

    return {
        "answer": response,
        "sources": sources,
        "time_taken_sec": time_taken,
        "mode_used": mode or "auto",  # ✅ expose mode to frontend
        "memory_turns": len(conversation_history),
    }
