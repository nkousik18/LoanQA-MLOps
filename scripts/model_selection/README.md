

#  **Model Selection & Evaluation Module**

**LoanDocQA+  Retrieval-Augmented Model Evaluation Framework**

This module implements the **end-to-end evaluation pipeline** used to benchmark multiple local LLMs (Ollama models) using RAG, intent routing, hallucination metrics, and structured scoring.
It is designed to test how well each LLM handles:

* Summarization
* Explanation
* Finance Q&A
* Retrieval-based answers
* Translation
* Hallucination avoidance
* Latency & performance

This is your **core model-evaluation system** before deploying LoanDocQA to production.

---
**LoanDocQA+ Retrieval-Augmented Model Evaluation Framework**  
Documentation (Google Docs): https://docs.google.com/document/d/1laSd_3Eb83f8EENUCYGbCuGDNnbXZJCvoZ0BDlmbGms/edit?usp=sharing
#  **Folder Overview**

```
model_selection/
│── chunking.py          # Hybrid semantic + sentence chunking
│── retriever.py         # GPU-accelerated MiniLM + Chroma vector DB
│── rag_pipeline.py      # RAG pipeline (router → retriever → prompt builder → LLM)
│── metrics.py           # Groundedness, severity, divergence, hallucination scoring
│── evaluate_by_intent.py# Main evaluation script (light & full modes)
│── llm_interface.py     # Ollama LLM interface (HTTP)
│── logger.py            # Logging utilities
│── visualize_results.py # Graphs: accuracy, hallucination, latency
```

---

#  **1. Setup Instructions**

## **Step 1  Ensure Python environment**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## **Step 2  Install required services**

### **Install Ollama**

```bash
brew install ollama
ollama serve   # or simply run: ollama run llama3.1
```

### **Pull the evaluation models**

```bash
ollama pull llama3.1
ollama pull phi3.5
ollama pull mistral
```

### **Install/update Chroma v0.5+**

```bash
pip install --upgrade chromadb
```

---

#  **2. Build the Vector Store**

Before running evaluation, you **must build the vector DB** from pre-cleaned OCR text.

```bash
python - << 'EOF'
from scripts.model_selection.retriever import MiniLMVectorStore

store = MiniLMVectorStore()
store.add_docs_from_folder("data/clean_texts")
EOF
```

Check vector count:

```bash
python - << 'EOF'
import chromadb
client = chromadb.PersistentClient(path="vector_db")
col = client.get_or_create_collection("loan_docs")
print("Vectors:", col.count())
EOF
```

If you see `> 0`, you're ready.

---

#  **3. Run LLM Evaluation**

You can benchmark **any model** available in Ollama.

## **Light Mode (8 queries, fast)**

Recommended for quick iteration.

```bash
python -m scripts.model_selection.evaluate_by_intent --model llama3.1 --light
```

## **Full Mode (40+ queries, all categories)**

Used for final submission-grade evaluation:

```bash
python -m scripts.model_selection.evaluate_by_intent --model llama3.1 --full
```

Replace `llama3.1` with:

```
phi3.5
mistral
qwen2.5
granite-code
wizard-math
```

---

#  **4. Where Results Are Stored**

All evaluation results are **automatically written to a CSV**:

```
evaluation_results/
   └── prompt_eval/
        └── summary.csv
```

This file contains:

* model name
* document name
* query
* intent predicted
* LLM answer
* groundedness score
* hallucination severity
* confidence score
* summary divergence
* hallucinated? yes/no
* pipeline latency
* llm latency


---

#  **5. Visualizing the Results**

Run:

```bash
python -m scripts.model_selection.visualize_results
```

This generates:

* hallucination distribution
* groundedness heatmap
* latency comparison
* intent accuracy
* model comparison bar charts

Outputs stored in:

```
evaluation_results/plots/
```

---

#  **6. Logs**

Every evaluation run generates structured logs:

```
logs/
   └── llm_logs/
   └── router_logs/
   └── retrieval_logs/
   └── rag_pipeline/
```

Router logs include:

```
[Router-HardRule] Q='...' → Intent=summary
[Router Decision] intent=summary conf=0.997 ctx_len=245
```

---




#  **8. Typical Evaluation Flow**

```
Query
   ↓
Intent Router
   ↓
Retriever (MiniLM + Chroma)
   ↓
Top-k context chunks
   ↓
PromptBuilder
   ↓
LLM (Ollama)
   ↓
Metrics (groundedness, hallucination, latency)
   ↓
Save to CSV
```

Everything is modular and can be swapped.

---


