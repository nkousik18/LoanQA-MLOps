
import os
import time
from datetime import datetime
from scripts.extraction_pipeline.config import setup_logger

# Initialize centralized logger
logger = setup_logger(__name__, log_type="extraction")

def save_text(output_text, output_dir, source_file):
    """
    Save extracted text to timestamped .txt file.
    Logs output file path and write duration.
    """
    start_time = time.time()
    base = os.path.splitext(os.path.basename(source_file))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"{base}_{timestamp}.txt")

    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(output_text)
        elapsed = time.time() - start_time
        logger.info(f"💾 Saved extracted text → {out_path} | Size: {len(output_text)} chars | Time: {elapsed:.2f}s")
        return out_path
    except Exception as e:
        logger.error(f"❌ Failed to save text file for {source_file}: {e}", exc_info=True)
        return None


def list_files(directory):
    """
    List all supported files (PDF + images).
    Logs file count and supported formats.
    """
    supported_exts = (".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp")
    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(supported_exts)
    ]
    logger.info(f"📂 Found {len(files)} supported files in {directory}")
    return files
