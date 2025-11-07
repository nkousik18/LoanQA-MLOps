# Document AI Full-Text & Layout Reconstruction (with Redaction)

Rebuilds a document's **full text and structure** (multi-column, tables, headers/footers) using **Google Document AI**, then **redacts PII** and writes:
- `<name>_document_full.txt`
- `<name>_document_structured.json`
- `<name>_document_preview.html`

## Quick Start
1. Enable APIs and create buckets (`scripts/bootstrap_gcp.sh`).
2. Create processors (`scripts/create_processors.sh`) and copy IDs into `src/config.py`.
3. Put a PDF into your input bucket.
4. Python env: `pip install -r requirements.txt`
5. Run: `python -m src.main`

See `src/*` for implementation details.
