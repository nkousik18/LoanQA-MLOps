from typing import Tuple
from google.cloud import storage
from .config import PROJECT_ID

def parse_gcs_uri(uri: str) -> Tuple[str, str]:
    assert uri.startswith("gs://")
    path = uri[5:]
    bkt, _, blob = path.partition("/")
    return bkt, blob

def read_gcs_bytes_and_type(uri: str) -> Tuple[bytes, str]:
    bkt, blob_name = parse_gcs_uri(uri)
    client = storage.Client(project=PROJECT_ID)
    blob = client.bucket(bkt).blob(blob_name)
    data = blob.download_as_bytes()
    ctype = blob.content_type or "application/octet-stream"
    return data, ctype

def write_gcs_bytes(uri: str, data: bytes, content_type: str) -> str:
    bkt, blob_name = parse_gcs_uri(uri)
    client = storage.Client(project=PROJECT_ID)
    blob = client.bucket(bkt).blob(blob_name)
    blob.upload_from_string(data, content_type=content_type)
    return uri
