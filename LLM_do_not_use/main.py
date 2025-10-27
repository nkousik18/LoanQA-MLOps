# LLMquery/main.py

import os
from text_preprocess import clean_text, chunk_text
from embeddings_index import build_vectorstore, load_vectorstore
from llm_interface import load_llm
from query_engine import create_query_engine, ask_question


def main():
    # Step 1: Load extracted text
    input_file = "data/loan_demo2_final.txt"
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"{input_file} not found.")

    with open(input_file, "r", encoding="utf-8") as f:
        raw_text = f.read()

    print("✅ Loaded extracted text from EasyOCR output.")

    # Step 2: Clean text
    cleaned = clean_text(raw_text)
    print("✅ Cleaned text.")

    # Step 3: Chunk text
    chunks = chunk_text(cleaned)
    print(f"✅ Split into {len(chunks)} chunks.")

    # Step 4: Build or Load FAISS vectorstore
    index_path = "LLMquery/vectorstores/loan_doc_index"
    if not os.path.exists(index_path):
        vectorstore = build_vectorstore(chunks, index_path)
        print("✅ Built new FAISS vector store.")
    else:
        vectorstore = load_vectorstore(index_path)
        print("✅ Loaded existing FAISS index.")

    # Step 5: Load local LLM
    llm = load_llm(model_name="llama3")
    print("✅ Loaded local LLM model.")

    # Step 6: Create Query Engine
    qa_chain = create_query_engine(llm, vectorstore)
    print("✅ Query engine ready.")

    # Step 7: Interactive Querying
    print("\n🔍 You can now query the document. Type 'exit' to quit.\n")
    while True:
        query = input("Your question: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Exiting query interface.")
            break
        answer = ask_question(qa_chain, query)
        print(f"\n💡 Answer:\n{answer}\n")


if __name__ == "__main__":
    main()
