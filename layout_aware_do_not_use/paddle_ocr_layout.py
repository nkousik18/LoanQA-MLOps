from paddleocr import PaddleOCR
from pathlib import Path
import json, os, warnings, platform

warnings.filterwarnings("ignore")

class PaddleLayoutExtractor:
    """
    Layout-aware text extractor using PaddleOCR.
    On macOS, lightweight models are enforced to avoid segfaults.
    """

    def __init__(self, lang="en", output_dir="out/paddleocr_layout"):
        self.lang = lang
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # --- stability flags ---
        os.environ["FLAGS_allocator_strategy"] = "naive_best_fit"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        # --- initialize PaddleOCR safely ---
        try:
            self.ocr = PaddleOCR(
                lang=self.lang,
                use_angle_cls=True,
                det_algorithm="DB",     # light detector
                rec_algorithm="CRNN",   # light recognizer
                det_model_dir=None,
                rec_model_dir=None,
            )
        except Exception as e:
            print(f"[⚠️] PaddleOCR init failed: {e}")
            raise RuntimeError(
                "PaddleOCR failed to initialize. Try `--method hybrid` on macOS."
            )

    def extract(self, pdf_path: str):
        print(f"[INFO] Running PaddleOCR on: {pdf_path}")
        try:
            results = self.ocr.ocr(pdf_path)
        except Exception as e:
            print(f"[❌] PaddleOCR crashed: {e}")
            raise RuntimeError(
                "PaddleOCR crashed (likely macOS segfault). Use --method hybrid instead."
            )

        output_json = []
        for page_id, page_data in enumerate(results):
            blocks = []
            for line in page_data:
                bbox, (text, conf) = line[0], line[1]
                blocks.append({"text": text, "confidence": conf, "bbox": bbox})
            output_json.append({"page": page_id + 1, "blocks": blocks})

        out_path = self.output_dir / f"{Path(pdf_path).stem}_paddle_layout.json"
        with open(out_path, "w") as f:
            json.dump(output_json, f, indent=2)
        print(f"[✅] Saved PaddleOCR layout output → {out_path}")
        return output_json
