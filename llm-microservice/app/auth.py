# app/auth.py

import time
import hmac
import hashlib
from fastapi import HTTPException

from app.config import settings


# ======================================================
# 1. Verify API Key
# ======================================================

def verify_api_key(api_key: str):
    """
    Simple header-based API key check.
    """
    if api_key is None:
        raise HTTPException(status_code=403, detail="API key missing")

    if api_key != settings.API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")


# ======================================================
# 2. Verify timestamp (replay protection)
# ======================================================

ALLOWED_DRIFT_SECONDS = 86400  # 1 minute window

def verify_timestamp(ts: str):
    """
    Timestamp must be recent to prevent replay attacks.
    """
    if ts is None:
        raise HTTPException(status_code=403, detail="Missing timestamp header")

    try:
        ts_int = int(ts)
    except ValueError:
        raise HTTPException(status_code=403, detail="Invalid timestamp format")

    now = int(time.time())
    if abs(now - ts_int) > ALLOWED_DRIFT_SECONDS:
        raise HTTPException(
            status_code=403,
            detail=f"Timestamp expired or too early (drift > {ALLOWED_DRIFT_SECONDS}s)"
        )


# ======================================================
# 3. Verify HMAC Signature
# ======================================================

def verify_signature(signature: str, ts: str, body: str):
    """
    Verify HMAC-SHA256 signature:

    signature = HMAC_SHA256(secret, timestamp + "." + body)

    The frontend must compute the same and send it as: X-Signature
    """
    if signature is None:
        raise HTTPException(status_code=403, detail="Missing signature header")

    if body is None:
        body = ""

    message = f"{ts}.{body}".encode()
    secret = settings.HMAC_SECRET.encode()

    # Produce server-side signature
    expected_sig = hmac.new(secret, message, hashlib.sha256).hexdigest()

    # Constant-time compare to prevent timing attacks
    if not hmac.compare_digest(expected_sig, signature):
        raise HTTPException(status_code=403, detail="Invalid signature")


# ======================================================
# 4. Helper: generate signature (optional)
#    Useful for debugging or offline testing
# ======================================================

def generate_signature(ts: str, body: str) -> str:
    """
    Helper function for debugging.
    """
    msg = f"{ts}.{body}".encode()
    return hmac.new(settings.HMAC_SECRET.encode(), msg, hashlib.sha256).hexdigest()
