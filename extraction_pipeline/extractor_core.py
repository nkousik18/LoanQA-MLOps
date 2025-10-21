import os
import fitz
from PIL import Image
from extraction_pipeline.ocr_utils import run_ocr_on_pdf_page, run_ocr_on_image
from extraction_pipeline.config import MIN_TEXT_LEN

def extract_text(file_path: str):
    """
    Extract text from any file: PDF or image.
    Automatically switches to OCR if needed.
    """
    ext = os.path.splitext(file_path)[-1].lower()

    if ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
        print(f"🖼️ Running OCR on image: {os.path.basename(file_path)}")
        return run_ocr_on_image(file_path)

    elif ext == ".pdf":
        print(f"📄 Extracting text from PDF: {os.path.basename(file_path)}")
        doc = fitz.open(file_path)
        full_text = []
        for i, page in enumerate(doc):
            text = page.get_text("text", flags=1 + 2 + 8)
            if len(text.strip()) < MIN_TEXT_LEN:
                print(f"[OCR fallback] Page {i+1} had low text density.")
                text = run_ocr_on_pdf_page(file_path, i)
            full_text.append(f"\n\n=== PAGE {i+1} ===\n{text.strip()}")
        return "\n".join(full_text)

    else:
        raise ValueError(f"Unsupported file type: {ext}")
