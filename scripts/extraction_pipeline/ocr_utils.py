"""
ocr_utils.py

Utility module for OCR extraction using PaddleOCR.

Implements lazy loading to prevent Airflow DAG parse crashes
caused by PaddleOCR initialization during DAG import.

Functions:
    get_ocr_engine()          -> Lazily initializes PaddleOCR once per worker
    run_ocr_on_image(path)    -> Runs OCR on image files (JPG, PNG)
    run_ocr_on_pdf_page(page) -> Runs OCR on PDF page using PyMuPDF
"""

import os
import fitz  # PyMuPDF
import logging
from functools import lru_cache
from PIL import Image
import io


# Lazy Load PaddleOCR
@lru_cache(maxsize=1)
def get_ocr_engine():
    """
    Lazily initializes the PaddleOCR model when first needed.
    Prevents Airflow DAG import failures and reduces cold-start latency.
    """
    from paddleocr import PaddleOCR
    logging.info("Initializing PaddleOCR engine lazily...")
    return PaddleOCR(use_angle_cls=True, lang="en")


# OCR for Image Files
def run_ocr_on_image(image_path: str):
    """
    Performs OCR on a given image file.

    Args:
        image_path (str): Path to image (PNG, JPG, etc.)
    Returns:
        text (str): Extracted text
    """
    try:
        ocr = get_ocr_engine()
        logging.info(f"Running OCR on image: {image_path}")
        result = ocr.ocr(image_path)
        text = "\n".join(
            [line[1][0] for block in result for line in block if len(line) > 1]
        )
        return text.strip()
    except Exception as e:
        logging.error(f"OCR on image failed: {e}")
        return ""


# OCR for PDF Pages
def run_ocr_on_pdf_page(pdf_path: str, page_number: int):
    """
    Extracts text from a given PDF page using PyMuPDF (fitz)
    and PaddleOCR for embedded images.

    Args:
        pdf_path (str): Path to PDF file
        page_number (int): Page index (0-based)
    Returns:
        text (str): OCR-extracted text
    """
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_number)
        pix = page.get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes))

        temp_img_path = f"/tmp/page_{page_number}.png"
        image.save(temp_img_path)

        ocr = get_ocr_engine()
        logging.info(f"Running OCR on PDF page {page_number} of {pdf_path}")
        result = ocr.ocr(temp_img_path)
        text = "\n".join(
            [line[1][0] for block in result for line in block if len(line) > 1]
        )

        os.remove(temp_img_path)
        return text.strip()
    except Exception as e:
        logging.error(f"OCR on PDF page {page_number} failed: {e}")
        return ""


# Debug Mode
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample = "/opt/airflow/data/sample_invoice.png"
    if os.path.exists(sample):
        text = run_ocr_on_image(sample)
        print("\nExtracted Text Preview:\n", text[:500])
    else:
        print("Sample image not found - skipping test run.")