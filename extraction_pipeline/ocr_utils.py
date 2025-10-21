import os
import fitz
import tempfile
from paddleocr import PaddleOCR
from PIL import Image

ocr_engine = PaddleOCR(use_angle_cls=True, lang="en")

def run_ocr_on_image(image_path):
    """Run OCR on a standalone image file."""
    result = ocr_engine.ocr(image_path)
    lines = []
    if result and result[0]:
        lines = [line[1][0] for line in result[0]]
    return "\n".join(lines)

def run_ocr_on_pdf_page(pdf_path, page_num):
    """Run OCR on a single page of a PDF."""
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    pix = page.get_pixmap()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
        pix.save(tmp_img.name)
        text = run_ocr_on_image(tmp_img.name)
        os.remove(tmp_img.name)
    return text
