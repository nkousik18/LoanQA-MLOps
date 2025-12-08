"""
config.py
----------
Central configuration for the AWS Extraction OCR → Textract → Normalization pipeline.

Now also supports writing outputs directly to GCS instead of local disk.

Key ideas:
- All data/log/report paths are defined as local *logical* Paths under PROJECT_ROOT
- When USE_GCS_OUTPUT=True, scripts should:
    - build a Path (e.g. RAW_DIR / "loan1_raw.json")
    - convert to a GCS object key via `to_gcs_key(path)`
    - read/write via google.cloud.storage (see scripts/aws_extraction_scripts/gcs_utils.py)
- WRITE_LOCAL_COPY controls whether a local copy is *also* kept on disk.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------
# PROJECT ROOT (local logical root)
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../doc-understand

# ---------------------------------------------------------------------
# OPTIONAL: GCS CREDENTIALS HELPER
# ---------------------------------------------------------------------
# If GOOGLE_APPLICATION_CREDENTIALS is already set (e.g. in Docker),
# we respect that. Otherwise, we look for a JSON key in gcp_keys/.
GCS_KEY_PATH = PROJECT_ROOT / "gcp_keys" / "mlops-loandoc-qa-4db11973be18.json"
if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ and GCS_KEY_PATH.exists():
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(GCS_KEY_PATH)

# ---------------------------------------------------------------------
# AWS SETTINGS (for Textract access to S3)
# ---------------------------------------------------------------------
BUCKET = os.getenv("S3_BUCKET", "textract-bucket-yash07")
REGION = os.getenv("AWS_REGION", "us-east-1")

# ---------------------------------------------------------------------
# GCS SETTINGS
# ---------------------------------------------------------------------
# Your main GCS bucket for everything (inputs + outputs)
GCS_BUCKET = os.getenv("GCS_BUCKET", "doc-understand-gcs-bucket-ash")

# If True → all data outputs should be considered stored in GCS.
# Scripts should:
#   - build a local logical Path (e.g. RAW_DIR / "loan1_raw.json")
#   - convert to key via to_gcs_key()
#   - use google.cloud.storage to read/write
USE_GCS_OUTPUT = True

# If True → ALSO keep a physical local file on disk
WRITE_LOCAL_COPY = False  # flip to True if you ever want both local+GCS

# ---------------------------------------------------------------------
# BASE DIRECTORIES (logical paths, mirror GCS layout)
# ---------------------------------------------------------------------
DATA_ROOT = PROJECT_ROOT / "data" / "aws_extraction_data"
LOGS_ROOT = PROJECT_ROOT / "logs" / "aws_extraction_logs"
REPORTS_ROOT = PROJECT_ROOT / "reports" / "aws_extraction_reports"

RAW_DIR = DATA_ROOT / "raw"
SEGMENTED_DIR = DATA_ROOT / "segmented"
NORMALIZED_DIR = DATA_ROOT / "normalized"
SCHEMA_DIR = DATA_ROOT / "schema"
RAW_TEXT_DIR = DATA_ROOT / "raw_text"
LAYOUT_DIR = DATA_ROOT / "layout_reconstructed"

# Live per-session pipeline (used by Streamlit/UI)
LIVE_PIPELINE_ROOT = PROJECT_ROOT / "data" / "local_pipeline"
LIVE_SESSIONS_DIR = LIVE_PIPELINE_ROOT / "sessions"

# Inputs you want to keep in GCS under data/
DOCS_DIR = PROJECT_ROOT / "data" / "docs"
USER_UPLOADS_DIR = PROJECT_ROOT / "data" / "user_uploads"

# ---------------------------------------------------------------------
# LOG DIRECTORY (local logs; you can later add a GCS handler if needed)
# ---------------------------------------------------------------------
AIRFLOW_HOME = Path("/opt/airflow")

if AIRFLOW_HOME.exists():
    LOG_DIR = AIRFLOW_HOME / "logs" / "aws_extraction_logs"
else:
    LOG_DIR = LOGS_ROOT

# ---------------------------------------------------------------------
# RUNTIME SETTINGS
# ---------------------------------------------------------------------
MAX_RETRIES = 5
RETRY_DELAY = 5  # seconds

# ---------------------------------------------------------------------
# Helpers for directories + GCS keys
# ---------------------------------------------------------------------
def ensure_directories() -> None:
    """
    Create local directories only if WRITE_LOCAL_COPY is True
    (for data outputs). Logs and reports will also follow this flag.

    When USE_GCS_OUTPUT=True and WRITE_LOCAL_COPY=False:
        - these directories are just *logical* and may never exist physically,
          which is fine as long as scripts don't rely on path.exists().
    """
    if not WRITE_LOCAL_COPY:
        return

    for d in [
        RAW_DIR,
        SEGMENTED_DIR,
        NORMALIZED_DIR,
        SCHEMA_DIR,
        LOG_DIR,
        REPORTS_ROOT,
        RAW_TEXT_DIR,
        LAYOUT_DIR,
        LIVE_SESSIONS_DIR,
        DOCS_DIR,
        USER_UPLOADS_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


def to_gcs_key(path: Path) -> str:
    """
    Convert a local logical path under PROJECT_ROOT into a GCS object key.

    Example:
      path = PROJECT_ROOT / "data" / "aws_extraction_data" / "raw" / "loan1_raw.json"
      -> "data/aws_extraction_data/raw/loan1_raw.json"
    """
    rel = path.relative_to(PROJECT_ROOT)
    return rel.as_posix()


def get_gcs_uri(path: Path) -> str:
    """
    Convert a local logical path to a full GCS URI.
    
    Example:
      path = PROJECT_ROOT / "data" / "aws_extraction_data" / "raw" / "loan1_raw.json"
      -> "gs://doc-understand-gcs-bucket/data/aws_extraction_data/raw/loan1_raw.json"
    """
    gcs_key = to_gcs_key(path)
    return f"gs://{GCS_BUCKET}/{gcs_key}"


def print_all_paths():
    print("\n" + "="*70)
    print("CONFIGURATION PATHS")
    print("="*70)
    print(f"Project Root (local) : {PROJECT_ROOT}")
    print(f"GCS Bucket           : {GCS_BUCKET}")
    print(f"USE_GCS_OUTPUT       : {USE_GCS_OUTPUT}")
    print(f"WRITE_LOCAL_COPY     : {WRITE_LOCAL_COPY}")
    
    if USE_GCS_OUTPUT:
        print(f"\n--- GCS Output Paths (Where Data Will Be Stored) ---")
        print(f"Raw Data             : {get_gcs_uri(RAW_DIR)}")
        print(f"Segmented            : {get_gcs_uri(SEGMENTED_DIR)}")
        print(f"Normalized           : {get_gcs_uri(NORMALIZED_DIR)}")
        print(f"Raw Text             : {get_gcs_uri(RAW_TEXT_DIR)}")
        print(f"Schema               : {get_gcs_uri(SCHEMA_DIR)}")
        print(f"Layout Recon         : {get_gcs_uri(LAYOUT_DIR)}")
        print(f"\n--- GCS Input Paths ---")
        print(f"Docs (batch)         : {get_gcs_uri(DOCS_DIR)}")
        print(f"User Uploads         : {get_gcs_uri(USER_UPLOADS_DIR)}")
        print(f"\n--- GCS Session Pipeline ---")
        print(f"Live Sessions        : {get_gcs_uri(LIVE_SESSIONS_DIR)}")
        print(f"\n--- GCS Logs & Reports ---")
        print(f"Logs                 : {get_gcs_uri(LOG_DIR)}")
        print(f"Reports              : {get_gcs_uri(REPORTS_ROOT)}")
    else:
        print(f"\n--- Local Output Paths ---")
        print(f"Data Root            : {DATA_ROOT}")
        print(f" ├── Raw Data        : {RAW_DIR}")
        print(f" ├── Segmented       : {SEGMENTED_DIR}")
        print(f" ├── Normalized      : {NORMALIZED_DIR}")
        print(f" ├── Raw Text        : {RAW_TEXT_DIR}")
        print(f" ├── Schema          : {SCHEMA_DIR}")
        print(f" └── Layout Recon    : {LAYOUT_DIR}")
        print(f"\n--- Local Input Paths ---")
        print(f"Docs (batch)         : {DOCS_DIR}")
        print(f"User Uploads         : {USER_UPLOADS_DIR}")
        print(f"\n--- Local Session Pipeline ---")
        print(f"Live Sessions        : {LIVE_SESSIONS_DIR}")
        print(f"\n--- Local Logs & Reports ---")
        print(f"Logs                 : {LOG_DIR}")
        print(f"Reports              : {REPORTS_ROOT}")
    
    print(f"\n--- AWS (for Textract) ---")
    print(f"S3 Bucket            : {BUCKET}")
    print(f"AWS Region           : {REGION}")
    print("="*70)
    if USE_GCS_OUTPUT:
        print("✅ ALL outputs will be written to GCS")
    else:
        print("✅ ALL outputs will be written locally")
    print("="*70 + "\n")


if __name__ == "__main__":
    ensure_directories()
    print_all_paths()
    print("✅ Configuration verified.")