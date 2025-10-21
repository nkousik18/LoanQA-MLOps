"""
inspect_index.py
----------------
Utility to visualize what's inside your Chroma vectorstore.
"""

import argparse
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import pandas as pd


def inspect_index(persist_dir: str, sample: int = 5):
    print(f"📂 Inspecting vectorstore at: {persist_dir}")

    # Load embeddings and Chroma DB
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    # Check if there are stored embeddings
    collection = db._collection
    count = collection.count()
    print(f"\n✅ Total embedded chunks: {count}")

    if count == 0:
        print("⚠️ No data found in this vectorstore.")
        return

    print(f"\n📄 Showing {sample} random entries:")
    results = collection.get(limit=sample)
    docs = results["documents"]
    metas = results["metadatas"]
    ids = results["ids"]

    rows = []
    for i in range(len(docs)):
        snippet = docs[i][:150].replace("\n", " ") + ("..." if len(docs[i]) > 150 else "")
        meta = metas[i] if metas and i < len(metas) else {}
        rows.append({
            "chunk_id": ids[i],
            "source": meta.get("source", "unknown"),
            "content": snippet
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    emb_dim = len(embeddings.embed_query("test"))
    print(f"\n🧠 Embedding Dimension: {emb_dim}")
    print("🔹 You can now query this vectorstore via your API or retriever.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a Chroma vectorstore.")
    parser.add_argument("--persist_dir", type=str, required=True, help="Path to the Chroma vectorstore directory.")
    parser.add_argument("--sample", type=int, default=5, help="Number of chunks to preview.")
    args = parser.parse_args()

    inspect_index(args.persist_dir, sample=args.sample)
