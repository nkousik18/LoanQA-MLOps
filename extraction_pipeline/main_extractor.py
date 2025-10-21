import os
from extraction_pipeline.extractor_core import extract_text
from extraction_pipeline.cleaner import clean_text
from extraction_pipeline.postprocessor import postprocess_text
from extraction_pipeline.utils import save_text, list_files
from extraction_pipeline.config import DATA_DIR, OUTPUT_DIR

def process_file(file_path):
    print(f"🔹 Processing: {os.path.basename(file_path)}")
    raw_text = extract_text(file_path)
    cleaned = clean_text(raw_text)
    final_text = postprocess_text(cleaned)
    save_text(final_text, OUTPUT_DIR, file_path)

def main():
    files = list_files(DATA_DIR)
    if not files:
        print("❌ No PDF or image files found in the data directory.")
        return
    for f in files:
        process_file(f)

if __name__ == "__main__":
    main()
