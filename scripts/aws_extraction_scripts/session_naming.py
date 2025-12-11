"""
session_naming.py
-----------------
Session naming and metadata helpers for multi-user document processing.

GCS-aware version: Checks both local and GCS for existing sessions.
"""

import os
import sys
import re
import uuid
import json
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------
# Ensure project root
# ---------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import from config + utils
from scripts.aws_extraction_scripts.config import (
    LIVE_SESSIONS_DIR,
    GCS_BUCKET,
    USE_GCS_OUTPUT,
    to_gcs_key,
)
from scripts.aws_extraction_scripts.gcs_utils import write_json
from scripts.aws_extraction_scripts.log_utils import get_logger

try:
    from google.cloud import storage
    HAS_GCS = True
except ImportError:
    HAS_GCS = False

# Use config's session directory
SESSIONS_ROOT = LIVE_SESSIONS_DIR
LOGGER = get_logger(__name__)


def clean_filename(name: str) -> str:
    """Sanitize filename to safe characters."""
    original = name
    name = name.lower().replace(" ", "_")
    cleaned = re.sub(r"[^a-zA-Z0-9._-]", "", name)
    if cleaned != original:
        LOGGER.debug(f"[session_naming] Cleaned filename '{original}' -> '{cleaned}'")
    return cleaned


def get_next_upload_counter(user_id: str) -> int:
    """
    Get the next upload counter for a user.
    Checks both local and GCS for existing sessions.
    """
    prefix = f"session_{user_id}_"
    counters = []

    LOGGER.info(f"[session_naming] Resolving next upload counter for user_id={user_id}")

    # Check local sessions
    if SESSIONS_ROOT.exists():
        for d in SESSIONS_ROOT.iterdir():
            if d.is_dir() and d.name.startswith(prefix):
                parts = d.name.split("_")
                if len(parts) >= 3:
                    try:
                        counters.append(int(parts[2]))
                    except Exception:
                        # ignore malformed names
                        continue

    # Check GCS sessions if enabled
    if USE_GCS_OUTPUT and HAS_GCS:
        try:
            client = storage.Client()
            gcs_prefix = to_gcs_key(SESSIONS_ROOT)
            if not gcs_prefix.endswith("/"):
                gcs_prefix += "/"

            LOGGER.debug(
                f"[session_naming] Listing GCS sessions in "
                f"bucket={GCS_BUCKET}, prefix={gcs_prefix}"
            )

            blobs = client.list_blobs(GCS_BUCKET, prefix=gcs_prefix, delimiter="/")

            for prefix_obj in blobs.prefixes:
                session_name = prefix_obj.rstrip("/").split("/")[-1]
                if session_name.startswith(prefix):
                    parts = session_name.split("_")
                    if len(parts) >= 3:
                        try:
                            counters.append(int(parts[2]))
                        except Exception:
                            continue
        except Exception as e:
            # GCS failure → local only fallback
            LOGGER.warning(
                f"[session_naming] Failed to list GCS sessions for user_id={user_id}: {e}"
            )

    next_counter = (max(counters) + 1) if counters else 1
    LOGGER.info(
        f"[session_naming] Next upload counter for user_id={user_id} -> {next_counter}"
    )
    return next_counter


def make_session_id(user_id: str, upload_counter: int) -> str:
    """Create structured session ID."""
    short_uid = uuid.uuid4().hex[:8]
    session_id = f"session_{user_id}_{upload_counter:03d}_{short_uid}"
    LOGGER.info(
        f"[session_naming] Created session_id={session_id} "
        f"for user_id={user_id}, upload_counter={upload_counter}"
    )
    return session_id


def write_session_metadata(session_path: Path, meta: dict) -> None:
    """
    Write session metadata to meta.json.
    GCS-aware: Uses gcs_utils so it goes to GCS when USE_GCS_OUTPUT=True
    """
    meta_path = session_path / "meta.json"
    LOGGER.info(f"[session_naming] Writing session metadata to {meta_path}")
    write_json(meta_path, meta)
