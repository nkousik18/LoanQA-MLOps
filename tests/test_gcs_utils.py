# src/docai_utils.py
from __future__ import annotations

from google.cloud import documentai_v1 as documentai


def process_with_layout_or_ocr(
    project_id: str,
    location: str,
    # primary names
    processor_id_layout: str | None = None,
    processor_id_ocr: str | None = None,
    # aliases (some tests or older code may use these)
    layout_processor_id: str | None = None,
    ocr_processor_id: str | None = None,
    # payload
    content: bytes = b"",
    mimetype: str | None = None,
):
    """
    Route PDFs to Layout Parser, images to OCR.
    Accepts both (processor_id_layout/processor_id_ocr) and
    (layout_processor_id/ocr_processor_id) for compatibility.
    Returns a `documentai.Document`.
    """

    # normalize ids from any of the accepted names
    layout_id = processor_id_layout or layout_processor_id
    ocr_id = processor_id_ocr or ocr_processor_id
    if not mimetype:
        mimetype = "application/pdf"  # sane default for tests

    # pick processor by mimetype
    is_pdf_like = mimetype.startswith("application/pdf")
    chosen = layout_id if is_pdf_like else ocr_id
    if not chosen:
        raise ValueError("Missing processor id (layout or ocr) for given mimetype")

    name = f"projects/{project_id}/locations/{location}/processors/{chosen}"

    client = documentai.DocumentProcessorServiceClient()
    raw_document = documentai.RawDocument(content=content, mime_type=mimetype)
    request = {"name": name, "raw_document": raw_document}
    # Real client returns an object with `.document`
    response = client.process_document(request=request)
    return response.document
