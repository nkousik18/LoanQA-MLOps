import argparse, platform
from paddle_ocr_layout import PaddleLayoutExtractor
from trocr_extractor import TrOCRExtractor
from hybrid_extractor import HybridExtractor

def main():
    parser = argparse.ArgumentParser(description="Layout-aware OCR Pipeline")
    parser.add_argument("--pdf", required=True, help="Input PDF path")
    parser.add_argument("--method", choices=["paddle", "trocr", "hybrid"],
                        default="paddle", help="OCR engine to use")
    args = parser.parse_args()

    # --- macOS safety switch ---
    system = platform.system()
    if system == "Darwin" and args.method == "paddle":
        print("[⚠️] PaddleOCR often crashes on macOS → switching to Hybrid mode.")
        args.method = "hybrid"

    if args.method == "paddle":
        extractor = PaddleLayoutExtractor()
    elif args.method == "trocr":
        extractor = TrOCRExtractor()
    elif args.method == "hybrid":
        extractor = HybridExtractor()
    else:
        raise ValueError("Invalid OCR method")

    extractor.extract(args.pdf)

if __name__ == "__main__":
    main()
