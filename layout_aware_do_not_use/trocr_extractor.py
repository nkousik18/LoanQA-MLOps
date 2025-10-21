import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import fitz
from pathlib import Path

class TrOCRExtractor:
    """Transformer-based OCR using Microsoft's TrOCR."""

    def __init__(self, model_name="microsoft/trocr-base-printed"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)

    def _pdf_to_images(self, pdf_path, dpi=300):
        pdf = fitz.open(pdf_path)
        paths = []
        for p in pdf:
            img_path = f"/tmp/page_{p.number}.png"
            p.get_pixmap(dpi=dpi).save(img_path)
            paths.append(img_path)
        return paths

    def extract(self, pdf_path: str):
        images = self._pdf_to_images(pdf_path)
        texts = []
        for path in images:
            image = Image.open(path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            ids = self.model.generate(**inputs)
            text = self.processor.batch_decode(ids, skip_special_tokens=True)[0]
            texts.append(text)
        out_path = Path(pdf_path).with_suffix(".trocr.txt")
        with open(out_path, "w") as f:
            f.write("\n\n".join(texts))
        print(f"[✅] TrOCR extracted text → {out_path}")
        return texts
