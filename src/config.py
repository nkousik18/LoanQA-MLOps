# Fill these before running
PROJECT_ID = "loan-doc-475418"
LOCATION   = "us"


# From `gcloud documentai processors list --location=us`
PROCESSOR_ID_LAYOUT = "8b59f6f331d7c18b"
PROCESSOR_ID_OCR    = "6355fe36456c06b6"  # optional; can be ""

# Input & Output
# If you set LOCAL_INPUT_FILE, it will read the local file instead of GCS.
LOCAL_INPUT_FILE = r"C:\Users\subha\PycharmProjects\doc\samples\loan1.pdf"  # e.g., "samples/loan.pdf" or leave empty to read from GCS
INPUT_URI = "" # gs://loan-doc-475418-docai-input/image.jpg
OUTPUT_BUCKET = "gs://loan-doc-475418-docai-redacted"

# Optional Cloud DLP for extra de-identification
USE_DLP = False
DLP_DEID_TEMPLATE = ""  # e.g., projects/<p>/locations/us/deidentifyTemplates/<id>

# Optional: KMS key for deterministic tokenization (not used directly in code; configure in DLP template)
KMS_KEY_NAME = ""
