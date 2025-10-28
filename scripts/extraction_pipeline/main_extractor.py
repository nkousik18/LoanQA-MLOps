# extraction_pipeline/main_extractor.py
from logging import Logger
# extraction_pipeline/main_extractor.py
from logging import Logger
import os
import traceback
import time
from scripts.extraction_pipeline.extractor_core import extract_text
from scripts.extraction_pipeline.utils import save_text, list_files
from scripts.extraction_pipeline.config import setup_logger, OUTPUT_DIR

# Initialize unified logger
logger = setup_logger(__name__, log_type="extraction")

def process_single_file(file_path: str):
    """
    Process a single document using extractor_core.
    Extract text and save to a unified directory: data/clean_texts.
    Gracefully skips files that cause OCR or Paddle runtime errors.
    """
    start_time = time.time()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # Use global OUTPUT_DIR from config.py
    output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Processing file: {file_path}")
    try:
        extracted_text = extract_text(file_path)
        if not extracted_text or len(extracted_text.strip()) == 0:
            raise ValueError("Empty text returned by extractor_core")

        out_path = save_text(extracted_text, output_dir, file_path)
        elapsed = time.time() - start_time
        logger.info(f"✅ Extracted text saved: {out_path} | Time taken: {elapsed:.2f}s")
        return out_path

    except Exception as e:
        logger.error(f"❌ Error processing {os.path.basename(file_path)}: {e}", exc_info=True)

        # 🩹 Optional fallback using pytesseract for image-based PDFs
        try:
            from PIL import Image
            import pytesseract
            if file_path.lower().endswith((".png", ".jpg", ".jpeg")):
                logger.warning(f"Attempting fallback OCR via pytesseract for {file_path}")
                text = pytesseract.image_to_string(Image.open(file_path))
                out_path = save_text(text, output_dir, file_path)
                logger.info(f"🟢 Fallback OCR successful → {out_path}")
                return out_path
        except Exception as fe:
            logger.warning(f"⚠️ Fallback OCR failed for {file_path}: {fe}", exc_info=True)

        return None


def run_extraction_pipeline(data_dir: str):
    """
    Batch process all documents in the given directory.
    Tracks number of successes, failures, and total runtime.
    """
    pipeline_start = time.time()
    logger.info(f"Starting extraction pipeline on directory: {data_dir}")

    files = list_files(data_dir)
    logger.info(f"Detected {len(files)} files for extraction: {files}")

    if not files:
        logger.warning("⚠️ No files found for extraction.")
        return

    success, failed = 0, 0
    for file_path in files:
        out_path = process_single_file(file_path)
        if out_path:
            success += 1
        else:
            failed += 1

    elapsed = time.time() - pipeline_start
    logger.info(f"📊 Extraction Summary — Success: {success}, Failed: {failed}, Duration: {elapsed:.2f}s")
    logger.info("[✅] Extraction pipeline run complete.")


def main(data_dir: str):
    """Alias for backward compatibility."""
    logger.info(f"Executing main() for directory: {data_dir}")
    return run_extraction_pipeline(data_dir)

if __name__ == "__main__":
    from scripts.extraction_pipeline.config import DATA_DIR
    logger.info(f"[ENTRYPOINT] Launching extraction pipeline on {DATA_DIR}")
    main(DATA_DIR)
