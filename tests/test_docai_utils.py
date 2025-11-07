# tests/test_docai_utils.py

# --- Build a full stub for google.cloud.documentai_v1 BEFORE importing the SUT ---
import sys, types

# Package skeleton
google = types.ModuleType("google")
cloud = types.ModuleType("google.cloud")
documentai_v1 = types.ModuleType("google.cloud.documentai_v1")

# Minimal "types" your code might touch at import/run time
class _DummyDoc:
    def __init__(self, text="OK"):
        self.text = text
        self.pages = []

# Some projects annotate return types as documentai.Document
# Provide that attribute so annotations/resolution at import time won’t crash
documentai_v1.Document = _DummyDoc

# If your code constructs these in requests, stub them as simple containers
class _RawDocument:
    def __init__(self, content=None, mime_type=None):
        self.content = content
        self.mime_type = mime_type

class _OcrConfig:
    def __init__(self, **kwargs):
        # accept any kwargs but do nothing
        for k, v in kwargs.items():
            setattr(self, k, v)

class _ProcessOptions:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

documentai_v1.RawDocument = _RawDocument
documentai_v1.OcrConfig = _OcrConfig
documentai_v1.ProcessOptions = _ProcessOptions

# Fake client returning an object with `.document`
from types import SimpleNamespace

class _FakeClient:
    def __init__(self, *a, **kw):
        self._last_name = None

    def process_document(self, request=None, name=None):
        # Return a structure that matches google’s: response.document
        self._last_name = name
        return SimpleNamespace(document=_DummyDoc(text="OK"))

# Factory class so code like documentai.DocumentProcessorServiceClient() works
class _FakeClientFactory:
    def __call__(self, *a, **kw):
        return _FakeClient()

documentai_v1.DocumentProcessorServiceClient = _FakeClientFactory()

# Register the modules so "from google.cloud import documentai_v1 as documentai" succeeds
google.cloud = cloud
sys.modules["google"] = google
sys.modules["google.cloud"] = cloud
sys.modules["google.cloud.documentai_v1"] = documentai_v1

# --- Now import the unit under test ---
import src.docai_utils as du
import pytest

def test_pdf_uses_layout():
    doc = du.process_with_layout_or_ocr(
        project_id="p",
        location="us",
        processor_id_layout="LAYOUT123",
        processor_id_ocr="OCR999",
        content=b"%PDF-1.7 ...",
        mimetype="application/pdf",
    )
    assert hasattr(doc, "text")
    assert isinstance(doc, documentai_v1.Document)

def test_image_uses_ocr():
    doc = du.process_with_layout_or_ocr(
        project_id="p",
        location="us",
        processor_id_layout="LAYOUT123",
        processor_id_ocr="OCR999",
        content=b"\xFF\xD8\xFF",
        mimetype="image/jpeg",
    )
    assert hasattr(doc, "text")
    assert isinstance(doc, documentai_v1.Document)
