# LLMquery/embeddings_index.py

import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# embeddings_index.py
# -------------------
# Loads the Chroma vectorstore built from text documents
# for use by the Loan Document Assistant API.

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def load_vectorstore(index_path="LLMquery/vectorstores/loan_doc_index"):
    """
    Loads a Chroma vectorstore and its embeddings.
    """
    print(f"📂 Loading Chroma vectorstore from: {index_path}")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db = Chroma(
        persist_directory=index_path,
        embedding_function=embeddings
    )

    print(f"✅ Loaded Chroma DB with {db._collection.count()} entries.")
    return db, embeddings



# embeddings_index.py
# -------------------
# Loads the Chroma vectorstore built from text documents
# for use by the Loan Document Assistant API.

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def load_vectorstore(index_path="LLMquery/vectorstores/loan_doc_index"):
    """
    Loads a Chroma vectorstore and its embeddings.
    """
    print(f"📂 Loading Chroma vectorstore from: {index_path}")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db = Chroma(
        persist_directory=index_path,
        embedding_function=embeddings
    )

    print(f"✅ Loaded Chroma DB with {db._collection.count()} entries.")
    return db, embeddings
