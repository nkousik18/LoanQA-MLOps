##  Environment Setup

### Dependencies

```bash
pip install langchain langchain-core langchain-community langchain-chroma langchain-huggingface chromadb>=1.0.21 sentence-transformers
```

You must also have:

* **Ollama** running locally.
* **Flask backend** active to serve API requests from the `/api` routes


## Directory Structure

```
scripts/LLMquery/                         # LLM and vector indexing logic
│   │   ├── build_index.py                # Builds and updates vectorstore (Chroma)
│   │   ├── query_index.py                # Retrieves context from embeddings
│   │   ├── prompts/                      # LLM prompt templates and routers
│   │   │   ├── prompt_router.py
│   │   │   ├── finance_prompts.py
│   │   │   ├── math_utils.py
│   │   │   ├── translate_prompt.py       # (New) prompt template for translations
│   │   │   └── __init__.py
│   │   └── __init__.py

```

---

## Key Components

###  **`build_index.py` — Document Vectorization & Indexing**

Handles creation, updating, and persistence of the **Chroma vectorstore** used for retrieval operations.

#### Features Implemented

* **Automatic vector index creation** from cleaned OCR text under `data/clean_texts/`.
* **Embedding generation** using SentenceTransformer model:

  ```
  sentence-transformers/all-MiniLM-L6-v2
  ```
* **Persistent storage** in `scripts/LLMquery/vectorstores/local_doc_index/`.
* **Automatic repair** of corrupted indices (removes and rebuilds).
* **Compatibility** with LangChain 0.3+ and Chroma 1.0+ (Rust backend).

#### Core Functions

```python
def add_to_index(new_file_path): 
    """Adds a single extracted text file to the existing vector index."""
```

* Loads `.txt` file → converts into LangChain `Document` object → generates embeddings → appends to vectorstore.

```python
def rebuild_vector_index():
    """Rebuilds the entire vectorstore from all clean_texts."""
```

* Iterates through all `.txt` files → reinitializes index → repopulates embeddings.

#### Configuration Parameters

```python
DATA_PATH   = "data/clean_texts"
INDEX_PATH  = "scripts/LLMquery/vectorstores/local_doc_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
```


####  Example Terminal Execution

```bash
python scripts/LLMquery/build_index.py
```

Expected log output:

```
 Rebuilding full vector index...
 Rebuilt Chroma index with 15 documents in 3.47s.
```



###  **`prompt_router.py` — Intelligent Prompt Routing**

Implements **semantic intent detection** for deciding which prompt template to apply to user queries or extracted text.
It uses sentence embeddings to classify inputs into 5 major intent types:

```
['finance', 'summary', 'translation', 'explanation', 'retrieval']
```

####  Integration with Flask

Used indirectly via:

```python
POST /api/summary
POST /api/translate
POST /api/explain
```

Each route calls the router internally to choose the proper LLM prompt and structure the query.

####  Example Output

```
 Prompt Router initialized with model: sentence-transformers/all-MiniLM-L6-v2
 Loaded 5 intent categories: ['finance', 'summary', 'translation', 'explanation', 'retrieval']
```


#### Customization

New templates can be easily added by defining new constants and registering them in `prompt_router.py`.


###  **`prompts/math_utils.py` — Lightweight Expression Evaluation**

Handles embedded mathematical reasoning or formulae detected in financial text.

#### Core Functions

```python
def evaluate_math(expression: str) -> float
```

* Parses and evaluates basic arithmetic safely (e.g., “12.5 * 3 + 7”).
* Used in combination with financial prompt routing to augment responses.


###  **`vectorstores/local_doc_index/` — Persistent Chroma Storage**

Stores all embeddings and metadata for processed documents.

* Automatically created and updated by `build_index.py`
* Contains:

  * `chroma.sqlite3`
  * `index/` folder with vector data
  * `config.json` (auto-managed by Chroma)

#### To reset:

```bash
rm -rf scripts/LLMquery/vectorstores/local_doc_index
python scripts/LLMquery/build_index.py
```

## Integration with Flask & Interface

LLMQuery serves as the **backend intelligence layer** that powers the `/api` endpoints and UI actions:

| Component          | Flask Endpoint   | Source             |
| ------------------ | ---------------- | ------------------ |
| Document Indexing  | `/api/upload`    | `build_index.py`   |
| Text Summarization | `/api/summary`   | `prompt_router.py` |
| Translation        | `/api/translate` | `prompt_router.py` |
| Explanation        | `/api/explain`   | `prompt_router.py` |

**Flow:**

```
1. User uploads a document → extracted text saved to data/clean_texts/
2. build_index.py indexes the file into Chroma
3. prompt_router.py classifies and routes LLM tasks
4. LLM response sent back to interface for display
```

---

##  Testing and Validation

### Run Unit Tests

```bash
pytest tests/test_LLMquery.py -v
```

Expected validations:

* Vector index creation and persistence
* LLM route classification correctness
* Chroma database read/write stability
* Math utility accuracy

### Log Monitoring

Check logs under:

```
logs/llm_logs/
```

Example:

```
llm_20251107_171822.log
```

Contains detailed tracing for embedding generation, index updates, and LLM response timing.
