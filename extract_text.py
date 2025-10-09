import os
import cv2
import json
from ingestion import pdf_to_images, preprocess_image
from ocr_engine import ocr_tesseract, ocr_easyocr
from layout_analysis import split_columns
from postprocess import clean_text
from glossary_utils import FinancialGlossary


# Initialize glossary (FinRAD + base financial glossary)
glossary = FinancialGlossary(
    glossary_path="data/financial_terms.csv",       # your curated glossary
    finrad_csv= "data/Finance_terms_definitions_labels.csv"            # FinRAD dataset (optional)
)


def save_output(text, file_path, suffix):
    """Helper to save output text to a file."""
    base, _ = os.path.splitext(file_path)
    out_file = f"{base}_{suffix}.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[INFO] Saved {suffix} output to: {out_file}")
    return out_file


def extract_text(file_path):
    """
    Main text extraction pipeline:
      - Converts PDF pages → images
      - Applies OCR (Tesseract / EasyOCR fallback)
      - Splits columns for correct reading order
      - Cleans text and detects glossary terms
    """

    ext = os.path.splitext(file_path)[1].lower()
    img_paths = pdf_to_images(file_path) if ext == ".pdf" else [file_path]
    all_text = []

    for img_path in img_paths:
        print(f"[INFO] Processing {img_path} ...")

        # Preprocess for OCR
        img = preprocess_image(img_path)
        columns = split_columns(img)
        page_texts = []

        for col_idx, col_img in enumerate(columns, start=1):
            temp_path = f"temp_col_{col_idx}.png"
            cv2.imwrite(temp_path, col_img)

            # OCR (Tesseract primary, EasyOCR fallback)
            text = ocr_tesseract(col_img)
            if len(text.strip()) < 50:
                print(f"[WARN] Weak OCR output in column {col_idx}, trying EasyOCR...")
                text = ocr_easyocr(temp_path)

            text = clean_text(text)
            page_texts.append(text)

        # Merge text from both columns
        page_text = "\n".join(page_texts)
        all_text.append(page_text)

    combined_text = "\n\n".join(all_text)
    save_output(combined_text, file_path, "final")

    # --- Glossary Integration ---
    found_terms = glossary.tag_terms(combined_text)
    definitions = {term: glossary.get_definition(term) for term in found_terms if glossary.get_definition(term)}

    glossary_json = os.path.splitext(file_path)[0] + "_glossary.json"
    with open(glossary_json, "w", encoding="utf-8") as f:
        json.dump(definitions, f, indent=4, ensure_ascii=False)
    print(f"[INFO] Saved glossary matches: {glossary_json}")
    print(f"[INFO] Detected {len(found_terms)} financial terms.\n")

    return combined_text


if __name__ == "__main__":
    # Example local test
    test_file = "data/federal-loan-programs.pdf"
    final_text = extract_text(test_file)

    print("\n--- Extracted Text Preview ---\n")
    print(final_text[:1000])
