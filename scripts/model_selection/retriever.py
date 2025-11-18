"""
GPU-accelerated MiniLM retriever using Chroma v0.5+
Clean + stable version
"""

import os
import torch
import chromadb
from sentence_transformers import SentenceTransformer
from scripts.model_selection.chunking import smart_chunk


# ------------------------------------------------------------
# Device Selection
# ------------------------------------------------------------
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = get_device()


# ------------------------------------------------------------
# Vector Store
# ------------------------------------------------------------
class MiniLMVectorStore:

    def __init__(self, db_path="vector_db", collection_name="loan_docs"):
        print(f"[Retriever] Using device: {DEVICE}")

        # embedder
        self.embedder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device=DEVICE,
        )
        print("[Retriever] MiniLM model loaded on:", DEVICE)

        # persistent chroma v0.5+
        self.client = chromadb.PersistentClient(path=db_path)

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # --------------------------------------------------------
    # Add documents (chunks)
    # --------------------------------------------------------
    def add_documents(self, docs):
        ids = [d["id"] for d in docs]
        texts = [d["text"] for d in docs]

        embeddings = self.embedder.encode(
            texts, convert_to_tensor=True, device=DEVICE, batch_size=32
        ).cpu().tolist()

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
        )

    # --------------------------------------------------------
    # Add from folder
    # --------------------------------------------------------
    def add_docs_from_folder(self, folder="data/clean_texts"):
        for fp in os.listdir(folder):
            if not fp.endswith(".txt"):
                continue

            doc_id = fp.replace(".txt", "")
            text = open(os.path.join(folder, fp)).read()

            chunks = smart_chunk(text)

            docs = [{"id": f"{doc_id}_{i}", "text": c} for i, c in enumerate(chunks)]
            self.add_documents(docs)

        print("[VectorStore] Index build complete.")

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------
    def search(self, query, k=5):
        q_emb = self.embedder.encode(
            query, convert_to_tensor=True, device=DEVICE
        ).cpu().tolist()

        result = self.collection.query(
            query_embeddings=[q_emb],
            n_results=k,
            include=["documents", "distances"]   # <-- FIX
        )

        # Chroma v0.5 returns lists-of-lists
        docs = []
        for i in range(len(result["ids"][0])):
            docs.append({
                "id": result["ids"][0][i],
                "text": result["documents"][0][i],
                "score": result["distances"][0][i],
            })

        return docs
