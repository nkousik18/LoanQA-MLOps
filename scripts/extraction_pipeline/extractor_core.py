# extraction_pipeline/extractor_core.py
from logging import Logger

import os
import time
import fitz
from PIL import Image
from scripts.extraction_pipeline.ocr_utils import run_ocr_on_pdf_page, run_ocr_on_image
from scripts.extraction_pipeline.config import setup_logger, MIN_TEXT_LEN


# Initialize logger for this module
logger = setup_logger(__name__, log_type="preprocessing")


def extract_text(file_path: str):
    """
    Extract text from any file: PDF or image.
    Automatically switches to OCR if needed.
    Logs each stage with timing, page count, and OCR backend information.
    """
    start_time = time.time()
    ext = os.path.splitext(file_path)[-1].lower()
    base_name = os.path.basename(file_path)

    try:
        if ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
            logger.info(f"🖼️ Running OCR on image: {base_name}")
            ocr_start = time.time()
            text = run_ocr_on_image(file_path)
            elapsed = time.time() - ocr_start
            logger.info(f"✅ OCR completed for image {base_name} in {elapsed:.2f}s")
            return text

        elif ext == ".pdf":
            logger.info(f"📄 Extracting text from PDF: {base_name}")
            doc = fitz.open(file_path)
            full_text = []
            total_pages = len(doc)
            logger.info(f"Detected {total_pages} pages in {base_name}")

            for i, page in enumerate(doc):
                page_num = i + 1
                page_start = time.time()

                # Extract text natively first
                text = page.get_text("text", flags=1 + 2 + 8)
                backend_used = "native text extraction"

                # Fallback to OCR for low text density pages
                if len(text.strip()) < MIN_TEXT_LEN:
                    logger.warning(f"[OCR fallback] Page {page_num} had low text density in {base_name}")
                    text = run_ocr_on_pdf_page(file_path, i)
                    backend_used = "OCR fallback"

                full_text.append(f"\n\n=== PAGE {page_num} ===\n{text.strip()}")
                page_elapsed = time.time() - page_start
                logger.debug(f"Page {page_num}/{total_pages} processed ({backend_used}) in {page_elapsed:.2f}s")

            total_elapsed = time.time() - start_time
            logger.info(f"✅ Extraction complete for {base_name} | Pages: {total_pages} | Time: {total_elapsed:.2f}s")
            return "\n".join(full_text)

        else:
            raise ValueError(f"Unsupported file type: {ext}")

    except Exception as e:
        logger.error(f"❌ Extraction failed for {base_name}: {str(e)}", exc_info=True)
        raise
