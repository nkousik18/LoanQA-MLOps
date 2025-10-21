from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import fitz
from pathlib import Path

class HybridExtractor:
    """Hybrid extractor combining DocTR (for layout) + PyMuPDF (for vector text)."""

    def __init__(self, use_gpu=False):
        self.model = ocr_predictor(pretrained=True)
        if use_gpu:
            self.model.cuda()

    def _extract_vector_text(self, pdf_path):
        doc = fitz.open(pdf_path)
        return "\n".join(page.get_text("text") for page in doc)

    def _extract_image_text(self, pdf_path):
        doc = DocumentFile.from_pdf(pdf_path)
        result = self.model(doc)
        exported = result.export()
        return "\n".join(
            blk["value"]
            for pg in exported["pages"]
            for blk in pg["blocks"]
        )

    def extract(self, pdf_path: str):
        print(f"[INFO] Running Hybrid (DocTR + PyMuPDF) on: {pdf_path}")
        text = self._extract_vector_text(pdf_path)
        img_text = self._extract_image_text(pdf_path)
        combined = text + "\n\n" + img_text
        out_path = Path(pdf_path).with_suffix(".hybrid.txt")
        with open(out_path, "w") as f:
            f.write(combined)
        print(f"[✅] Hybrid extracted text → {out_path}")
        return combined
