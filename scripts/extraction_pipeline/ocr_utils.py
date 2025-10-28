# extraction_pipeline/ocr_utils.py

import os
import fitz
import tempfile
import time
from paddleocr import PaddleOCR
from PIL import Image
from scripts.extraction_pipeline.config import setup_logger

# Initialize centralized logger
logger = setup_logger(__name__, log_type="preprocessing")

# Initialize OCR engine once
ocr_engine = PaddleOCR(use_angle_cls=True, lang="en")

def run_ocr_on_image(image_path):
    """
    Run OCR on a standalone image file.
    Logs backend details, confidence scores, and runtime.
    """
    start_time = time.time()
    logger.info(f"🖼️ Running OCR on image: {os.path.basename(image_path)} using PaddleOCR")

    try:
        result = ocr_engine.ocr(image_path)
        lines, confidences = [], []
        if result and result[0]:
            for line in result[0]:
                text, conf = line[1][0], line[1][1]
                lines.append(text)
                confidences.append(conf)

        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        elapsed = time.time() - start_time
        logger.info(f"✅ OCR complete for {os.path.basename(image_path)} | "
                    f"Lines: {len(lines)} | Avg confidence: {avg_conf:.2f} | Time: {elapsed:.2f}s")
        return "\n".join(lines)

    except Exception as e:
        logger.error(f"❌ OCR failed for image {image_path}: {e}", exc_info=True)
        return ""


def run_ocr_on_pdf_page(pdf_path, page_num):
    """
    Run OCR on a single page of a PDF.
    Converts page to image and runs OCR while tracking timing and results.
    """
    start_time = time.time()
    base_name = os.path.basename(pdf_path)

    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
            pix.save(tmp_img.name)
            text = run_ocr_on_image(tmp_img.name)
        os.remove(tmp_img.name)

        elapsed = time.time() - start_time
        logger.info(f"📄 OCR on {base_name} page {page_num + 1} completed in {elapsed:.2f}s | "
                    f"Chars extracted: {len(text)}")
        return text

    except Exception as e:
        logger.error(f"❌ OCR failed for {base_name} page {page_num + 1}: {e}", exc_info=True)
        return ""
