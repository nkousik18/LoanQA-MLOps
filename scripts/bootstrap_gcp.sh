#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${1:-$(gcloud config get-value project)}"
INPUT_BUCKET="${2:-$PROJECT_ID-docai-input}"
OUTPUT_BUCKET="${3:-$PROJECT_ID-docai-redacted}"

gcloud services enable documentai.googleapis.com storage.googleapis.com dlp.googleapis.com

gsutil mb -l us gs://$INPUT_BUCKET || true
gsutil mb -l us gs://$OUTPUT_BUCKET || true

gsutil uniformbucketlevelaccess set on gs://$INPUT_BUCKET
gsutil uniformbucketlevelaccess set on gs://$OUTPUT_BUCKET

cat > lifecycle.json << 'JSON'
{
  "rule": [
    { "action": {"type": "Delete"}, "condition": {"age": 2} }
  ]
}
JSON
gsutil lifecycle set lifecycle.json gs://$INPUT_BUCKET
rm lifecycle.json

echo "Done. Buckets:"
echo "  gs://$INPUT_BUCKET"
echo "  gs://$OUTPUT_BUCKET"
