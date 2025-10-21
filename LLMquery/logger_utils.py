# LLMquery/logger_utils.py

import csv
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer, util

# Load lightweight embedding model for similarity scoring
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def compute_confidence(question, context):
    """
    Compute cosine similarity between question and retrieved context
    to estimate grounding confidence (0–1 scale).
    """
    if not context.strip():
        return 0.0
    q_emb = embedding_model.encode(question, convert_to_tensor=True)
    c_emb = embedding_model.encode(context, convert_to_tensor=True)
    score = util.pytorch_cos_sim(q_emb, c_emb).item()
    return round(score, 3)

def log_query(question, answer, sources, confidence, logfile="LLMquery/query_log.csv"):
    """
    Append each query-answer interaction to a CSV log.
    """
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(logfile, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, question, answer, sources, confidence])
