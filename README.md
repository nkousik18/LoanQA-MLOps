# LoanQA-MLOps

## doc-understand

## Structure
- data/raw: original PDFs + Textract/DocAI JSON
- data/segmented: line-level spans (pre-cleanup)
- data/normalized: cleaned spans + stats + validation outputs
- outputs: logs, manifest, reports
- scripts: utility + extraction + LLM query scripts
- dags: Airflow DAGs
- tests: unit tests

## Quick Start
1. Place PDFs in `data/raw/`
2. Run OCR → store JSON in `data/raw/<doc_id>_ocr.json`
3. Segment the document → `data/segmented/`
4. Normalize and clean → `data/normalized/`
5. Run validation → generates reports in `data/normalized/`
