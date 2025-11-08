# doc-understand

## Structure
- data/raw: original PDFs + Textract JSON
- data/segmented: line-level spans (before cleanup)
- data/normalized: cleaned/filtered spans + page stats + validation
- outputs: logs, manifest, reports
- scripts: utility scripts
- dags: Airflow DAGs
- tests: unit tests

## Quick start
1) Put PDFs in data/raw/.
2) Run OCR → save JSON to data/raw/<doc_id>_textract.json.
3) Convert/segment → data/segmented/.
4) Normalize → data/normalized/.
5) Validate → write a validation report in data/normalized/.

