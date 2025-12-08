# retriever.py
import io
import json
import numpy as np
from functools import lru_cache
from google.cloud import storage
import torch
from sentence_transformers import SentenceTransformer, util
from app.utils import log


class CloudRetriever:
    """
    Loads a GLOBAL vectorstore from:
        gs://<bucket>/data/local_pipeline/vectorstore/

    Expected files:
        chunks.json
        embeddings.npy
    """

    def __init__(
        self,
        bucket_name="doc-understand-gcs-bucket-ash",
        prefix="data/local_pipeline/vectorstore"
    ):
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.prefix = prefix

        log.info("[Retriever] Initializing embedder (MiniLM-L6-v2)")
        self.embedder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.device = self.embedder.device
        log.info(f"[Retriever] Embedder running on {self.device}")

    # ---------------------------------------------------------
    # LOAD VECTORSTORE (CACHED IN MEMORY)
    # ---------------------------------------------------------
    @lru_cache(maxsize=1)
    def load_vectorstore(self):
        """Loads chunks.json + embeddings.npy once."""
        log.info(
            f"[Retriever] Loading vectorstore from gs://{self.bucket.name}/{self.prefix}"
        )

        # -------------------------------
        # Load chunks.json
        # -------------------------------
        chunks_blob = self.bucket.blob(f"{self.prefix}/chunks.json")
        if not chunks_blob.exists():
            raise FileNotFoundError(
                f"chunks.json not found at {self.prefix}/"
            )

        chunks = json.loads(chunks_blob.download_as_text())

        # -------------------------------
        # Load embeddings.npy
        # -------------------------------
        emb_blob = self.bucket.blob(f"{self.prefix}/chunk_embeddings.npy")
        if not emb_blob.exists():
            raise FileNotFoundError(
                f"embeddings.npy not found at {self.prefix}/"
            )

        emb_bytes = emb_blob.download_as_bytes()
        embeddings_np = np.load(io.BytesIO(emb_bytes))

        # Convert to torch tensor on SAME DEVICE as embedder
        embeddings = torch.tensor(
            embeddings_np,
            dtype=torch.float32,
            device=self.device
        )

        log.info(
            f"[Retriever] Loaded {len(chunks)} chunks | "
            f"Embeddings shape = {embeddings.shape} on {self.device}"
        )

        return chunks, embeddings

    # ---------------------------------------------------------
    # RETRIEVAL LOGIC
    # ---------------------------------------------------------
    def retrieve(self, question: str, top_k: int = 3):
        """Retrieve best-matching chunks using cosine similarity."""

        try:
            chunks, embeddings = self.load_vectorstore()
        except Exception as e:
            log.error(f"[Retriever] Failed to load vectorstore: {e}")
            return []

        # Encode query → same device as embeddings
        query_vec = self.embedder.encode(
            question, convert_to_tensor=True
        ).to(self.device)

        scores = util.cos_sim(query_vec, embeddings)[0]

        # Get top-K highest similarity scores
        top_idx = scores.topk(top_k).indices.tolist()

        results = []
        for idx in top_idx:
            results.append({
                "text": chunks[idx]["text"],
                "metadata": chunks[idx].get("metadata", {}),
                "score": float(scores[idx])
            })

        log.info(
            f"[Retriever] Retrieved {len(results)} chunks | "
            f"Top score = {results[0]['score'] if results else None}"
        )

        return results


# Global instance
retriever = CloudRetriever()
