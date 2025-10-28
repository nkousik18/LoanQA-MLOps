# extraction_pipeline/postprocessor.py

import re
import time
from scripts.extraction_pipeline.config import setup_logger

# Initialize centralized logger
logger = setup_logger(__name__, log_type="extraction")

def postprocess_text(text: str):
    """
    Normalize bullets, spacing, and capitalization.
    Logs all normalization actions and timing.
    """
    start_time = time.time()
    logger.debug("Starting text postprocessing and normalization.")

    try:
        initial_len = len(text)
        text = re.sub(r"[\t|•@#_=~]", "", text)
        text = re.sub(r"\n[-–—]\s*", "\n• ", text)
        text = re.sub(r"(?<!\.)\n(?=[a-z])", " ", text)
        text = re.sub(r"\s{2,}", " ", text)
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)  # separate glued words
        text = text.strip()

        elapsed = time.time() - start_time
        logger.info(f"✅ Postprocessing complete | Input length: {initial_len} | "
                    f"Output length: {len(text)} | Time: {elapsed:.2f}s")
        return text

    except Exception as e:
        logger.error(f"❌ Postprocessing failed: {e}", exc_info=True)
        return text
