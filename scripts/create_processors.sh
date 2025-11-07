#!/usr/bin/env bash
set -euo pipefail
PROJECT_ID="${1:-$(gcloud config get-value project)}"

gcloud documentai processors create \
  --project=$PROJECT_ID --location=us \
  --display-name="layout-parser" \
  --type=LAYOUT_PARSER_PROCESSOR

gcloud documentai processors create \
  --project=$PROJECT_ID --location=us \
  --display-name="ocr" \
  --type=OCR_PROCESSOR

gcloud documentai processors list --project=$PROJECT_ID --location=us
