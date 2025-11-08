"""
run_pipeline.py
---------------
Orchestrates the full document-understanding pipeline:
1️⃣ Fetch PDFs from S3
2️⃣ Run Textract OCR
3️⃣ Segment text lines
4️⃣ Normalize cleaned text

Each stage logs outputs and saves intermediate JSON files in:
  data/raw → data/segmented → data/normalized
"""

import os, sys, time
from datetime import datetime

# ---------------------------------------------------------------------
# 🔧 Ensure working directory is project root
# ---------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if os.path.basename(PROJECT_ROOT).lower() != "doc-understand":
    PROJECT_ROOT = os.path.join(PROJECT_ROOT, "doc-understand")
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print("📂 Running from:", os.getcwd())

# ---------------------------------------------------------------------
# 📦 Imports
# ---------------------------------------------------------------------
from scripts.aws_extraction_scripts.log_utils import get_logger
from scripts.aws_extraction_scripts.fetch_files import fetch_files
from scripts.aws_extraction_scripts.run_textract import run_textract_all
from scripts.aws_extraction_scripts.segment_text import run_segmentation_all
from scripts.aws_extraction_scripts.normalize_text import run_normalization_all
from scripts.aws_extraction_scripts.tracker import track_task

logger = get_logger("run_pipeline")

# ---------------------------------------------------------------------
# 🧾 Helper: Log summary
# ---------------------------------------------------------------------
def log_pipeline_summary(summary: dict):
    """Appends a summary of pipeline execution to logs/pipeline_summary.log"""
    log_dir = os.path.join("logs")
    os.makedirs(log_dir, exist_ok=True)
    summary_path = os.path.join(log_dir, "pipeline_summary.log")

    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(f"{json.dumps(summary, indent=2)}\n{'='*50}\n")

    logger.info(f"📜 Pipeline summary logged → {summary_path}")


# ---------------------------------------------------------------------
# ⚙️ Pipeline Orchestrator
# ---------------------------------------------------------------------
def run_full_pipeline():
    """Executes all pipeline stages sequentially."""
    start_time = time.time()
    logger.info("🚀 Starting full document-understanding pipeline...\n")

    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stages": {},
        "status": "IN_PROGRESS"
    }

    try:
        # 1️⃣ Fetch PDFs
        logger.info("📥 Stage 1: Fetching PDFs from S3...")
        pdfs = fetch_files()
        summary["stages"]["fetch_files"] = len(pdfs)
        if not pdfs:
            msg = "⚠️ No PDFs found in S3 bucket."
            logger.warning(msg)
            summary["status"] = "NO_FILES"
            log_pipeline_summary(summary)
            return

        # 2️⃣ Run Textract OCR
        stage_start = time.time()
        logger.info("🔍 Stage 2: Running Textract OCR...")
        textract_outputs = run_textract_all()
        summary["stages"]["run_textract"] = len(textract_outputs)
        logger.info(f"⏱️ Textract stage duration: {time.time() - stage_start:.2f}s")

        # 🔁 NEW LOGIC — continue even if Textract skipped everything
        if not textract_outputs:
            logger.info("⚡ No new Textract files generated — continuing with existing raw data.")

        # 3️⃣ Segment Text Lines
        stage_start = time.time()
        logger.info("✂️ Stage 3: Segmenting text lines...")
        segmented_outputs = run_segmentation_all()
        summary["stages"]["segment_text"] = len(segmented_outputs)
        logger.info(f"⏱️ Segmentation duration: {time.time() - stage_start:.2f}s")

        if not segmented_outputs:
            logger.info("⚠️ No new segmented files — continuing with previous outputs.")

        # 4️⃣ Normalize Text
        stage_start = time.time()
        logger.info("🧹 Stage 4: Normalizing segmented text...")
        normalized_outputs = run_normalization_all()
        summary["stages"]["normalize_text"] = len(normalized_outputs)
        logger.info(f"⏱️ Normalization duration: {time.time() - stage_start:.2f}s")

        if not normalized_outputs:
            logger.warning("⚠️ No new normalized files generated.")
        else:
            logger.info(f"✅ Normalized {len(normalized_outputs)} files successfully.")

        # ✅ Final summary
        elapsed = time.time() - start_time
        summary["duration_sec"] = round(elapsed, 2)
        summary["status"] = "SUCCESS"
        logger.info(f"🎯 Pipeline completed in {elapsed:.2f} seconds.")
        logger.info("🏁 All stages executed successfully.")
        log_pipeline_summary(summary)

    except Exception as e:
        summary["status"] = "FAILED"
        summary["error"] = str(e)
        log_pipeline_summary(summary)
        logger.exception(f"❌ Pipeline failed: {e}")
        raise


# ---------------------------------------------------------------------
# 🧩 Run Individual Stages (Optional CLI)
# ---------------------------------------------------------------------
def run_stage(stage_name):
    """Run a specific stage manually."""
    stage_name = stage_name.lower()
    if stage_name == "fetch":
        return fetch_files()
    elif stage_name == "textract":
        return run_textract_all()
    elif stage_name == "segment":
        return run_segmentation_all()
    elif stage_name == "normalize":
        return run_normalization_all()
    else:
        logger.error(f"❌ Unknown stage: {stage_name}")
        return None


# ---------------------------------------------------------------------
# 🏁 Entry Point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Run full or partial document pipeline.")
    parser.add_argument(
        "--stage",
        type=str,
        choices=["fetch", "textract", "segment", "normalize", "full"],
        default="full",
        help="Which stage to run (default: full pipeline).",
    )
    args = parser.parse_args()

    if args.stage == "full":
        run_full_pipeline()
    else:
        logger.info(f"⚙️ Running single stage: {args.stage}")
        run_stage(args.stage)
