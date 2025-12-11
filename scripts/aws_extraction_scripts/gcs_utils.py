import os
import sys
import json
from pathlib import Path
from typing import Any, Optional

from google.cloud import storage

# ---------------------------------------------------------------------
# Ensure project root on sys.path, then import config + logger
# ---------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))  # .../doc-understand

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.aws_extraction_scripts.config import (
    GCS_BUCKET,
    USE_GCS_OUTPUT,
    WRITE_LOCAL_COPY,
    to_gcs_key,
)

from scripts.aws_extraction_scripts.log_utils import get_logger

LOGGER = get_logger(__name__)

# ---------------------------------------------------------------------
# Internal: client + blob helpers
# ---------------------------------------------------------------------
_GCS_CLIENT: Optional[storage.Client] = None


def get_gcs_client() -> storage.Client:
    """
    Lazily create and reuse a single storage.Client.
    Respects GOOGLE_APPLICATION_CREDENTIALS set in config.py or env.
    """
    global _GCS_CLIENT
    if _GCS_CLIENT is None:
        LOGGER.info("[gcs_utils] Creating new GCS client")
        _GCS_CLIENT = storage.Client()
    return _GCS_CLIENT


def _get_blob_for_path(path: Path) -> storage.Blob:
    """
    Given a logical path under PROJECT_ROOT, return its GCS Blob.
    """
    client = get_gcs_client()
    key = to_gcs_key(path)
    LOGGER.debug(f"[gcs_utils] Resolving blob for key={key}")
    bucket = client.bucket(GCS_BUCKET)
    return bucket.blob(key)


# ---------------------------------------------------------------------
# Existence checks
# ---------------------------------------------------------------------
def gcs_exists(path: Path) -> bool:
    """Check if a blob exists in GCS for the given logical path."""
    if not USE_GCS_OUTPUT:
        return False
    blob = _get_blob_for_path(path)
    exists = blob.exists()
    LOGGER.debug(f"[gcs_utils] gcs_exists={exists} path={path}")
    return exists


def local_exists(path: Path) -> bool:
    """Check if a local file exists for the given logical path."""
    exists = path.exists()
    LOGGER.debug(f"[gcs_utils] local_exists={exists} path={path}")
    return exists


def logical_exists(path: Path) -> bool:
    """
    Combined existence check:
    - If USE_GCS_OUTPUT: check GCS (primary) OR local (if present)
    - Else: check local only
    """
    if USE_GCS_OUTPUT:
        exists = gcs_exists(path) or local_exists(path)
    else:
        exists = local_exists(path)
    LOGGER.debug(f"[gcs_utils] logical_exists={exists} path={path}")
    return exists


# ---------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------
def write_bytes(path: Path, data: bytes, content_type: Optional[str] = None) -> None:
    """
    Write binary data to GCS and/or local, based on config flags.

    - path: logical path under PROJECT_ROOT
    - data: bytes to write
    - content_type: optional MIME type for GCS
    """
    # Local copy (if enabled or if GCS disabled)
    if WRITE_LOCAL_COPY or not USE_GCS_OUTPUT:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        LOGGER.info(f"[gcs_utils] Wrote local bytes: {path}")

    # GCS copy (if enabled)
    if USE_GCS_OUTPUT:
        blob = _get_blob_for_path(path)
        if content_type:
            blob.upload_from_string(data, content_type=content_type)
        else:
            blob.upload_from_string(data)
        LOGGER.info(f"[gcs_utils] Uploaded bytes to GCS: {path} (bucket={GCS_BUCKET})")


def write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """
    Write text content to GCS and/or local.

    - Encoded as UTF-8 by default.
    """
    data = text.encode(encoding)
    write_bytes(path, data, content_type="text/plain; charset=utf-8")


def write_json(path: Path, obj: Any, indent: int = 2) -> None:
    """
    Serialize obj as JSON and write to GCS and/or local.

    - Uses utf-8 encoding.
    """
    text = json.dumps(obj, indent=indent, ensure_ascii=False)
    write_text(path, text)


# ---------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------
def read_bytes(path: Path) -> bytes:
    """
    Read binary data from GCS or local.

    Priority:
    - If USE_GCS_OUTPUT: try GCS first; if missing, fall back to local.
    - Else: read from local only.

    Raises FileNotFoundError if not found anywhere.
    """
    # Primary: GCS
    if USE_GCS_OUTPUT:
        blob = _get_blob_for_path(path)
        if blob.exists():
            LOGGER.info(f"[gcs_utils] Reading bytes from GCS: {path}")
            return blob.download_as_bytes()

    # Fallback: local
    if path.exists():
        LOGGER.info(f"[gcs_utils] Reading bytes from local: {path}")
        return path.read_bytes()

    LOGGER.error(f"[gcs_utils] File not found in GCS or local: {path}")
    raise FileNotFoundError(f"File not found in GCS or local: {path}")


def read_text(path: Path, encoding: str = "utf-8") -> str:
    """
    Read text content (UTF-8) from GCS or local.
    """
    data = read_bytes(path)
    return data.decode(encoding)


def read_json(path: Path) -> Any:
    """
    Read JSON file from GCS or local and parse it.
    """
    text = read_text(path)
    return json.loads(text)


# ---------------------------------------------------------------------
# Convenience: upload/download whole files
# ---------------------------------------------------------------------
def upload_local_file(local_path: Path, logical_target_path: Path, content_type: Optional[str] = None) -> None:
    """
    Upload a local file (local_path) to GCS at the given logical_target_path.
    Optionally also keep/overwrite a local copy at logical_target_path if WRITE_LOCAL_COPY is True.

    Example:
        local_path = Path("/tmp/loan1_raw.json")
        logical_target_path = RAW_DIR / "loan1_raw.json"
        upload_local_file(local_path, logical_target_path)
    """
    LOGGER.info(
        f"[gcs_utils] Uploading local file {local_path} -> logical {logical_target_path}"
    )
    data = local_path.read_bytes()
    write_bytes(logical_target_path, data, content_type=content_type)


def download_to_local(path: Path, local_target: Optional[Path] = None) -> Path:
    """
    Download from GCS to a local file.

    - path: logical path under PROJECT_ROOT (where the file lives in GCS)
    - local_target: where to save locally.
        * If None -> save to 'path' itself under PROJECT_ROOT.

    Returns:
        The Path of the local file.
    """
    if local_target is None:
        local_target = path

    LOGGER.info(
        f"[gcs_utils] Downloading {path} -> local {local_target}"
    )
    data = read_bytes(path)  # this handles GCS vs local
    local_target.parent.mkdir(parents=True, exist_ok=True)
    local_target.write_bytes(data)
    return local_target


# ---------------------------------------------------------------------
# Debug helper
# ---------------------------------------------------------------------
def debug_print_storage_mode() -> None:
    """
    Print a one-line summary of current storage behavior.
    """
    mode = []
    if USE_GCS_OUTPUT:
        mode.append("GCS")
    if WRITE_LOCAL_COPY:
        mode.append("LocalCopy")
    if not mode:
        mode.append("LocalOnly")

    msg = f"[gcs_utils] Storage mode: {', '.join(mode)} (bucket={GCS_BUCKET})"
    print(msg)
    LOGGER.info(msg)


if __name__ == "__main__":
    # Quick self-check when running directly
    debug_print_storage_mode()
