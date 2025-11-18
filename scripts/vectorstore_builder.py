"""
Vectorstore Builder for LoanDocAI
---------------------------------
Creates:
 - chunks.json      (text chunks)
 - embeddings.npy   (MiniLM vectors)
 - index.faiss      (FAISS-GPU/CPU index)

This script runs locally (Mac/Windows) and produces files
that the inference microservice will load in RunPod.

"""

import os
import json
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------

DATA_DIR = "data/clean_texts"
OUT_DIR = "vectorstore"
CHUNK_SIZE = 350        # optimal for MiniLM
CHUNK_OVERLAP = 50      # small overlap

os.makedirs(OUT_DIR, exist_ok=True)

CHUNKS_FILE = f"{OUT_DIR}/chunks.json"
EMB_FILE = f"{OUT_DIR}/embeddings.npy"
FAISS_FILE = f"{OUT_DIR}/index.faiss"


# ----------------------------------------------------------
# DEVICE SELECTION
# ----------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = get_device()
print(f"[Builder] Using device: {DEVICE}")


# ----------------------------------------------------------
# LOAD EMBEDDING MODEL
# ----------------------------------------------------------

print("[Builder] Loading MiniLM embedding model...")
embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device=DEVICE
)


# ----------------------------------------------------------
# CHUNKING FUNCTION
# ----------------------------------------------------------

def chunk_text(text: str, chunk_size=350, overlap=50):
    tokens = text.split()
    chunks = []

    i = 0
    while i < len(tokens):
        chunk = tokens[i:i+chunk_size]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap

    return chunks


# ----------------------------------------------------------
# STEP 1: READ & CHUNK DOCUMENTS
# ----------------------------------------------------------

all_chunks = []
counter = 0

print("[Builder] Loading and chunking documents...")

for fp in os.listdir(DATA_DIR):
    if not fp.endswith(".txt"):
        continue

    doc_id = fp.replace(".txt", "")
    path = os.path.join(DATA_DIR, fp)

    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    doc_chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

    for c in doc_chunks:
        all_chunks.append({
            "id": counter,
            "doc_id": doc_id,
            "text": c,
        })
        counter += 1

print(f"[Builder] Total chunks: {len(all_chunks)}")

# Save chunks
with open(CHUNKS_FILE, "w") as f:
    json.dump(all_chunks, f, indent=2)

print(f"[✓] Saved: {CHUNKS_FILE}")


# ----------------------------------------------------------
# STEP 2: EMBEDDINGS
# ----------------------------------------------------------

print("[Builder] Computing embeddings...")

texts = [c["text"] for c in all_chunks]

# batching improves speed
batch_size = 64
embeddings = []

for i in tqdm(range(0, len(texts), batch_size)):
    batch = texts[i:i+batch_size]
    emb = embedder.encode(batch, convert_to_numpy=True, device=DEVICE)
    embeddings.append(emb)

embeddings = np.vstack(embeddings)

# Save embeddings
np.save(EMB_FILE, embeddings)
print(f"[✓] Saved: {EMB_FILE}")


# ----------------------------------------------------------
# STEP 3: BUILD FAISS INDEX
# ----------------------------------------------------------

dim = embeddings.shape[1]
print(f"[Builder] Creating FAISS index of dimension {dim}...")

# GPU INDEX IF AVAILABLE
if torch.cuda.is_available():
    print("[Builder] Using FAISS-GPU index...")
    res = faiss.StandardGpuResources()
    index = faiss.IndexFlatL2(dim)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    gpu_index.add(embeddings)
    index = faiss.index_gpu_to_cpu(gpu_index)
else:
    print("[Builder] Using FAISS-CPU index...")
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

faiss.write_index(index, FAISS_FILE)
print(f"[✓] Saved: {FAISS_FILE}")

print("\n[✓] Vectorstore build complete.")
print("Upload these three files to GCP / S3 for inference service:")
print(" • chunks.json")
print(" • embeddings.npy")
print(" • index.faiss")
