"""
build_index.py
Smart vectorstore builder — adds only new documents if index already exists.
Usage:
  python -m LLMquery.build_index --input_dir data/loan_docs --persist_dir LLMquery/vectorstores/loan_doc_index
"""

import os
import argparse
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# ============================================================
# 1️⃣ Parse CLI arguments
# ============================================================
parser = argparse.ArgumentParser(description="Build or update Chroma vectorstore for loan documents")
parser.add_argument("--input_dir", type=str, required=True, help="Directory containing .txt documents")
parser.add_argument("--persist_dir", type=str, required=True, help="Directory to store vector index")
args = parser.parse_args()

input_dir = args.input_dir
persist_dir = args.persist_dir

# ============================================================
# 2️⃣ Initialize embeddings and vectorstore
# ============================================================
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load existing Chroma DB if available
if os.path.exists(os.path.join(persist_dir, "chroma.sqlite3")):
    print(f"📂 Existing Chroma index found at: {persist_dir}")
    db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    existing_sources = set(
        [m["source"] for m in db.get(include=["metadatas"])["metadatas"] if "source" in m]
    )
else:
    print(f"🆕 No existing Chroma index found. Creating new one at: {persist_dir}")
    db = None
    existing_sources = set()

# ============================================================
# 3️⃣ Load and split new documents
# ============================================================
print(f"📑 Scanning input folder: {input_dir}")
new_files = []
for file in os.listdir(input_dir):
    if file.endswith(".txt") and file not in existing_sources:
        new_files.append(os.path.join(input_dir, file))

if not new_files:
    print("✅ No new documents found. Vectorstore is already up to date.")
    exit()

print(f"🆕 Found {len(new_files)} new file(s): {new_files}")

docs = []
for path in new_files:
    loader = TextLoader(path, encoding="utf-8")
    documents = loader.load()
    for d in documents:
        d.metadata["source"] = os.path.basename(path)
    docs.extend(documents)

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = splitter.split_documents(docs)

print(f"✂️ Split into {len(split_docs)} text chunks.")

# ============================================================
# 4️⃣ Add to Chroma vectorstore
# ============================================================
if db is None:
    db = Chroma.from_documents(split_docs, embeddings, persist_directory=persist_dir)
else:
    db.add_documents(split_docs)

db.persist()
print(f"✅ Index updated successfully with {len(split_docs)} new chunks.")
print(f"📦 Total entries in Chroma DB: {db._collection.count()}")
