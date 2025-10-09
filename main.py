import argparse
from extract_text import extract_text

def main():
    parser = argparse.ArgumentParser(description="Mini Document AI - Text Extraction Pipeline")
    parser.add_argument(
        "--file",
        required=True,
        help="Path to the input file (PDF or image)"
    )
    args = parser.parse_args()

    print("\n==============================")
    print("🚀 Mini Document AI Pipeline")
    print("==============================\n")

    file_path = args.file
    print(f"[INFO] Processing file: {file_path}\n")

    extracted_text = extract_text(file_path)

    print("\n==============================")
    print("✅ Extraction Complete!")
    print("==============================\n")
    print(extracted_text[:800])  # preview the first 800 characters

if __name__ == "__main__":
    main()
