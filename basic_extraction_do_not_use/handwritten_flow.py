"""
handwritten_extractor.py
------------------------
Augments the existing OCR + glossary system to handle handwritten fields
without disturbing your existing pipeline.
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Load TrOCR (small for Apple Silicon)
MODEL_NAME = "microsoft/trocr-small-handwritten"
processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


# ---------- STEP 1: find likely handwriting zones ----------
def detect_handwritten_regions(image):
    """
    Find filled boxes / handwritten areas.
    Returns list of (x,y,w,h) bounding boxes.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # remove printed lines
    binv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV, 31, 9)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    lines = cv2.morphologyEx(binv, cv2.MORPH_OPEN, h_kernel) + cv2.morphologyEx(binv, cv2.MORPH_OPEN, v_kernel)
    handwriting_mask = cv2.subtract(binv, lines)
    handwriting_mask = cv2.medianBlur(handwriting_mask, 3)

    contours, _ = cv2.findContours(handwriting_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    H, W = gray.shape
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if 30 < w < 0.5*W and 15 < h < 0.25*H:  # ignore noise
            boxes.append((x, y, w, h))
    return boxes


# ---------- STEP 2: recognize handwriting ----------
def recognize_handwriting(image, boxes):
    results = []
    for (x, y, w, h) in boxes:
        crop = image[y:y+h, x:x+w]
        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        pixel_values = processor(pil, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            ids = model.generate(pixel_values, max_length=64)
        text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        if text:
            results.append({"bbox": [x, y, w, h], "text": text})
    return results


# ---------- STEP 3: detect tick/check marks ----------
def detect_checkmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ticks = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if 8 < w < 25 and 8 < h < 25 and 0.75 < w/h < 1.25:
            roi = gray[y:y+h, x:x+w]
            fill = (roi < 128).sum() / (w*h)
            if fill > 0.25:
                ticks.append({"bbox": [x, y, w, h], "checked": True})
    return ticks


# ---------- STEP 4: main entry ----------
def extract_handwritten_content(input_path):
    """
    Combines printed OCR results (already done) with handwritten detections.
    Returns dict: {'handwritten_text': [...], 'checkboxes': [...]}.
    """
    import easyocr
    reader = easyocr.Reader(['en'])
    image = cv2.imread(str(input_path))
    printed = reader.readtext(image, detail=1, paragraph=True)

    printed_text = []
    for item in printed:
        if len(item) == 3:
            bbox, text, conf = item
            if conf > 0.5:
                printed_text.append({"bbox": bbox, "text": text, "confidence": conf})
        elif len(item) == 2:
            bbox, text = item
            printed_text.append({"bbox": bbox, "text": text, "confidence": None})

    boxes = detect_handwritten_regions(image)
    handwriting = recognize_handwriting(image, boxes)
    checkboxes = detect_checkmarks(image)

    return {"printed_text": printed_text, "handwritten_text": handwriting, "checkboxes": checkboxes}


# ---------- STEP 5: CLI ----------
if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(description="Extract handwritten content alongside printed OCR")
    ap.add_argument("--file", "-f", required=True, help="Input scanned form image/PDF")
    ap.add_argument("--out", "-o", default="data/extracted/handwritten_output.json", help="Output JSON path")
    args = ap.parse_args()

    result = extract_handwritten_content(Path(args.file))
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"✅ Saved merged OCR+handwriting output → {args.out}")
