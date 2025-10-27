# extraction_pipeline/main_extractor.py

import os
import traceback
from extraction_pipeline.extractor_core import extract_text
from extraction_pipeline.utils import save_text, list_files

def process_single_file(file_path: str):
    """
    Process a single document using extractor_core.
    Extract text and save to a unified directory: data/clean_texts.
    Gracefully skips files that cause OCR or Paddle runtime errors.
    """
    print(f"\n[INFO] Processing single file: {file_path}")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.join(project_root, "data", "clean_texts")
    os.makedirs(output_dir, exist_ok=True)

    try:
        extracted_text = extract_text(file_path)
        if not extracted_text or len(extracted_text.strip()) == 0:
            raise ValueError("Empty text returned by extractor_core")

        out_path = save_text(extracted_text, output_dir, file_path)
        print(f"[✅] Extracted text saved to: {out_path}")
        return out_path

    except Exception as e:
        print(f"[❌] Error processing {os.path.basename(file_path)}: {e}")
        traceback.print_exc()

        # 🩹 Optional fallback using pytesseract for image-based PDFs
        try:
            from PIL import Image
            import pytesseract
            if file_path.lower().endswith((".png", ".jpg", ".jpeg")):
                print("[🔄] Trying fallback OCR via pytesseract...")
                text = pytesseract.image_to_string(Image.open(file_path))
                out_path = save_text(text, output_dir, file_path)
                print(f"[🟢] Fallback OCR successful → {out_path}")
                return out_path
        except Exception as fe:
            print(f"[⚠️] Fallback OCR also failed for {file_path}: {fe}")

        # Return None so Airflow can skip gracefully
        return None


def run_extraction_pipeline(data_dir: str):
    """
    Batch process all documents in the given directory.
    """
    print(f"[INFO] Running extraction pipeline on: {data_dir}")
    files = list_files(data_dir)
    if not files:
        print("⚠️ No files found for extraction.")
        return

    success, failed = 0, 0
    for file_path in files:
        out_path = process_single_file(file_path)
        if out_path:
            success += 1
        else:
            failed += 1

    print(f"\n[📊] Extraction Summary: {success} succeeded, {failed} failed.")
    print("[✅] Pipeline run complete.")


def main(data_dir: str):
    """Alias for backward compatibility."""
    return run_extraction_pipeline(data_dir)
