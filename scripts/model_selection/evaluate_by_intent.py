"""
Evaluation Pipeline (Cross-Platform GPU, Light/Full Modes)
Runs:
 - Intent detection
 - Prompt routing
 - RAG pipeline
 - Hallucination scoring (groundedness, severity, confidence, divergence)
 - Summary CSV logging
"""

import os
import json
import time
import argparse
import pandas as pd

from scripts.model_selection.rag_pipeline import RAGPipeline
from scripts.model_selection.retriever import MiniLMVectorStore
from scripts.model_selection.metrics import detect_hallucination
from scripts.LLMquery.prompts.prompt_router import build_prompt
from scripts.model_selection.logger import log_event


# ============================================================
# Evaluation Query Sets (FULL and LIGHT)
# ============================================================

EVAL_QUERIES_FULL = {
    "summary": [
        "Summarize the main differences between Direct Subsidized, Unsubsidized, and PLUS loans.",
        "Give a concise overview of why federal student loans are recommended over private loans.",
        "Summarize the key repayment terms in this personal loan agreement.",
        "Provide a high-level overview of borrower and lender obligations described in this agreement.",
        "Summarize the primary themes described in this document.",
        "Provide a concise overview of the ethical considerations discussed."
    ],
    "explanation": [
        "Explain what ‘Direct Subsidized Loan’ means in this document.",
        "Explain the difference between a federal student loan and a private student loan.",
        "Explain what ‘Rule of 78’ refers to in the repayment section.",
        "What does ‘jointly and severally liable’ mean in this agreement?"
    ],
    "finance": [
        "What interest rate applies to Direct PLUS Loans disbursed after July 1, 2022?",
        "Based on the context, what interest rate applies to graduate Direct Unsubsidized Loans?",
        "Does the agreement specify any actual interest rate value?",
        "According to the contract, how is the late payment fee calculated?"
    ],
    "retrieval": [
        "Who is eligible for Direct PLUS Loans according to this document?",
        "When are you NOT charged interest on a Direct Subsidized Loan?",
        "According to this agreement, what happens if the borrower fails to make a payment?",
        "What collateral or security is used in this loan agreement?"
    ],
    "translation": [
        "Translate: ‘fixed interest rate’ into Spanish.",
        "Translate: ‘income-based repayment plan’ into Hindi.",
        "Translate this phrase into Telugu: ‘borrower is in default’.",
        "Translate this clause into Spanish: ‘jointly and severally liable’."
    ]
}

# ---------------- Light Mode ---------------- #

EVAL_QUERIES_LIGHT = {
    "summary": [
        "Summarize the main differences between Direct Subsidized and Unsubsidized loans."
    ],
    "explanation": [
        "Explain what a Direct Subsidized Loan means in this document."
    ],
    "finance": [
        "What interest rate applies to Direct PLUS Loans disbursed after July 1, 2022?",
        "Does this loan agreement specify the actual interest rate?"
    ],
    "retrieval": [
        "Who is eligible for Direct PLUS Loans according to this document?",
        "Does the document mention when repayment begins?"
    ],
    "translation": [
        "Translate 'fixed interest rate' into Spanish.",
        "Translate this into Hindi: borrower responsibilities."
    ]
}


# ============================================================
# Load Documents
# ============================================================

def load_clean_texts(folder="data/clean_texts"):
    docs = {}
    for fp in os.listdir(folder):
        if fp.endswith(".txt"):
            with open(os.path.join(folder, fp), "r") as f:
                docs[fp.replace(".txt", "")] = f.read()
    return docs


# ============================================================
# Run Evaluation for a Single Model
# ============================================================

def evaluate_model(model_name, mode="light"):
    """
    Runs evaluation over all documents.
    Saves results to evaluation_results/prompt_eval/summary.csv.
    """

    print(f"\n[RUN] Evaluating model: {model_name} (mode={mode})\n")

    out_dir = "evaluation_results/prompt_eval"
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "summary.csv")

    # load existing results if present
    df_existing = pd.read_csv(out_csv) if os.path.exists(out_csv) else pd.DataFrame()

    # choose evaluation set
    queries = EVAL_QUERIES_LIGHT if mode == "light" else EVAL_QUERIES_FULL

    # initialize RAG + retrieve
    store = MiniLMVectorStore()
    rag = RAGPipeline(index_store=store)

    docs = load_clean_texts()

    all_rows = []

    for doc_id, doc_text in docs.items():
        print(f"[DOC] Evaluating: {doc_id}")

        for intent_category, q_list in queries.items():

            for q in q_list:
                start = time.time()

                # -------------------------------
                # Step 1 — RAG Pipeline
                # -------------------------------
                result = rag.run(model_name, q, doc_text)

                # -------------------------------
                # Step 2 — Hallucination scoring
                # -------------------------------
                scores = detect_hallucination(
                    result["llm_output"],
                    result["retrieved_chunks"]
                )

                # -------------------------------
                # Step 3 — Store structured row
                # -------------------------------
                row = {
                    "timestamp": time.time(),
                    "model": model_name,
                    "doc": doc_id,

                    # routing & query
                    "intent_group": intent_category,
                    "router_intent": result["intent"],
                    "query": q,

                    # output
                    "llm_output": result["llm_output"],

                    # hallucination metrics
                    "groundedness": scores["groundedness"],
                    "severity": scores["severity"],
                    "confidence": scores["confidence"],
                    "summary_divergence": scores["summary_divergence"],
                    "hallucinated": scores["hallucinated"],
                    "verdict": scores["verdict"],

                    # performance
                    "llm_latency": result["llm_latency"],
                    "pipeline_latency": result["pipeline_latency"],
                }

                all_rows.append(row)

    # -------------------------------
    # Save CSV
    # -------------------------------
    df_new = pd.DataFrame(all_rows)
    df_out = pd.concat([df_existing, df_new], ignore_index=True)
    df_out.to_csv(out_csv, index=False)

    print(f"\n[✓] Evaluation complete.")
    print(f"[✓] Saved summary at: {out_csv}")
    return df_out


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--light", action="store_true")

    args = parser.parse_args()
    mode = "full" if args.full else "light"

    evaluate_model(args.model, mode)
