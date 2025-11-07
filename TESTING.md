# TESTING

### Unit Tests (pytest)
We run isolated unit tests for redaction, ordering, table reconstruction, rendering, GCS helpers, and DocAI routing.

- Local run: `pytest --cov=src --cov-report=term-missing`
- No network calls are made; external clients are monkeypatched/stubbed.
- For convenience, the test environment either installs `google-cloud-documentai` / `google-cloud-storage` or stubs `google.cloud.documentai_v1` directly in `tests/test_docai_utils.py`.


## Scope
Unit tests validate preprocessing (redaction), ordering, tables, render, and GCS/DocAI routing without cloud calls.

## How to run
```bash
python -m venv .venv && . .venv/Scripts/activate  # Windows PowerShell
pip install -r requirements.txt -r requirements-test.txt
pytest --cov=src --cov-report=term-missing
