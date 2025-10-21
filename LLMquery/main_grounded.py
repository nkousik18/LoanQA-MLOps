"""
main_grounded.py
Dynamic, streaming-capable, fully grounded QA pipeline for loan documents.
Now runs on a local LLM (no OpenAI API).
Features:
- Semantic re-ranking (no hardcoded keywords)
- Real-time token streaming
- Confidence scoring
- Source metadata in JSON output
"""

import numpy as np
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA

# Choose one local model integration:
# Option A: Ollama
from langchain_community.llms import Ollama

# Option B (if you use local HuggingFace Transformer model)
# from langchain_community.llms import HuggingFacePipeline
# from transformers import pipeline


# ====================================================
# 1️⃣  Load Vectorstore & Embeddings
# ====================================================

def load_vectorstore(index_path="LLMquery/vectorstores/loan_doc_index"):
    """
    Loads your persisted Chroma vector database and MiniLM embeddings.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory=index_path, embedding_function=embeddings)
    return db, embeddings


# ====================================================
# 2️⃣  Dynamic Semantic Re-ranking
# ====================================================

def dynamic_rerank(question, docs, embeddings, top_k=5):
    """
    Re-rank retrieved documents based on cosine similarity between
    the question and each chunk embedding.
    Returns (top_docs, confidence_score)
    """
    q_emb = np.array(embeddings.embed_query(question))
    scores = []
    for d in docs:
        d_emb = np.array(embeddings.embed_query(d.page_content[:1000]))
        sim = np.dot(q_emb, d_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(d_emb))
        scores.append(sim)

    ranked = [d for _, d in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]
    avg_conf = float(np.mean(sorted(scores, reverse=True)[:top_k])) if scores else 0.0
    return ranked[:top_k], round(avg_conf, 3)


# ====================================================
# 3️⃣  Prompt Template
# ====================================================

qa_prompt = PromptTemplate(
    input_variables=["question", "context"],
    template=(
        "You are a precise assistant. Use ONLY information found in the context.\n\n"
        "Question: {question}\n\n"
        "Context:\n{context}\n\n"
        "Rules:\n"
        "- Quote or closely paraphrase what is written in the context; do NOT invent or combine topics.\n"
        "- Keep the answer to a maximum of 3 short sentences.\n"
        "- Do not include deferment, consolidation, or forgiveness unless those exact words appear in the context.\n"
        "- If no explicit answer exists, reply exactly: 'Information not available in the provided document.'\n"
        "- Always end with (Source: StudentAid.gov/repay)\n\n"
        "Answer:"
    ),
)




# ====================================================
# 4️⃣  Local LLM (Streaming Enabled)
# ====================================================

# LLMquery/llm_interface.py

from langchain.callbacks.base import BaseCallbackHandler

class PrintStreamHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token, **kwargs):
        print(token, end="", flush=True)

def create_local_llm(model_name="mistral", temperature=0):
    from langchain_community.llms import Ollama
    llm = Ollama(
        model=model_name,
        temperature=temperature,
        streaming=True,
        callbacks=[PrintStreamHandler()],
    )
    return llm



# Example for HuggingFacePipeline (comment above Ollama and uncomment below):
# def create_local_llm(model_name="mistralai/Mistral-7B-Instruct-v0.2", temperature=0):
#     pipe = pipeline(
#         "text-generation",
#         model=model_name,
#         max_new_tokens=512,
#         temperature=temperature,
#         device_map="auto"
#     )
#     return HuggingFacePipeline(pipeline=pipe)


# ====================================================
# 5️⃣  QA Pipeline (Retrieval → Rerank → Local LLM Generation)
# ====================================================

def answer_question(question, index_path="LLMquery/vectorstores/loan_doc_index", model_name="mistral"):
    db, embeddings = load_vectorstore(index_path)
    retriever = db.as_retriever(search_kwargs={"k": 12})
    llm = create_local_llm(model_name=model_name)

    print(f"\n[🔎] Retrieving context for: \"{question}\" ...")
    docs = retriever.get_relevant_documents(question)

    filtered_docs, confidence = dynamic_rerank(question, docs, embeddings, top_k=5)
    context_text = "\n\n".join(d.page_content for d in filtered_docs)

    print(f"[ℹ️] Retrieved {len(filtered_docs)} chunks | Confidence ≈ {confidence}")

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff",
        chain_type_kwargs={"prompt": qa_prompt},
    )

    print("\n[💬] Generating answer (streaming below):\n")
    result = chain({"query": question, "context": context_text})

    # Collect metadata
    src_metadata = []
    for d in result.get("source_documents", []):
        meta = {
            "source": d.metadata.get("source", ""),
            "page": d.metadata.get("page", ""),
            "chunk_id": d.metadata.get("chunk_id", ""),
        }
        src_metadata.append(meta)

    print("\n\n[✅] Final Answer Complete.")
    return {
        "question": question,
        "answer": result["result"],
        "confidence": confidence,
        "sources": src_metadata,
    }


# ====================================================
# 6️⃣  Run Example
# ====================================================

if __name__ == "__main__":
    question = "How to resolve loan problems quickly?"
    response = answer_question(question, model_name="mistral")

    print("\n───────────────────────────────")
    print(f"[QUESTION] {response['question']}")
    print(f"[CONFIDENCE] {response['confidence']}")
    print(f"[SOURCES]\n{response['sources']}")
    print("───────────────────────────────\n")
