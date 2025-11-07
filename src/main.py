import pathlib, mimetypes, json
from google.cloud import documentai_v1 as documentai

from .config import (
    PROJECT_ID, LOCATION, PROCESSOR_ID_LAYOUT, PROCESSOR_ID_OCR,
    LOCAL_INPUT_FILE, INPUT_URI, OUTPUT_BUCKET, USE_DLP, DLP_DEID_TEMPLATE
)
from .gcs_utils import read_gcs_bytes_and_type, write_gcs_bytes, parse_gcs_uri
from .docai_utils import process_document_with_processor
from .ordering import reconstruct_boxes
from .tables import tables_for_doc
from .render import render_markdown_html
from .redact import regex_redact, assert_clean


def _rotate(img, angle_deg):
    import cv2, numpy as np
    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def _deskew(binary_img):
    import cv2, numpy as np
    coords = cv2.findNonZero(255 - binary_img)
    if coords is None:
        return binary_img
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    return _rotate(binary_img, angle)

def preprocess_image_for_ocr(img_bytes: bytes) -> bytes:
    import cv2, numpy as np
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    target_min = 1800
    scale = max(1.0, target_min / max(h, w))
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.bilateralFilter(gray, 7, 60, 60)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)
    bw = _deskew(bw)

    try:
        cv2.imwrite("debug_preprocessed.jpg", bw)
    except Exception:
        pass

    ok, enc = cv2.imencode(".jpg", bw, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return enc.tobytes() if ok else img_bytes


def load_source() -> tuple[bytes, str, str]:
    if LOCAL_INPUT_FILE:
        with open(LOCAL_INPUT_FILE, "rb") as f:
            content = f.read()
        print(f"[DEBUG] Loaded {len(content)} bytes from {LOCAL_INPUT_FILE}")
        mt = mimetypes.guess_type(LOCAL_INPUT_FILE)[0] or "application/octet-stream"
        return content, mt, pathlib.Path(LOCAL_INPUT_FILE).name
    else:
        content, mt = read_gcs_bytes_and_type(INPUT_URI)
        _, blob_name = parse_gcs_uri(INPUT_URI)
        return content, mt, blob_name.rsplit("/",1)[-1]


def main():
    content, mimetype, source_id = load_source()
    print(f"Source: {source_id} ({mimetype})")

    # --- Try layout processor first ---
    doc = process_document_with_processor(
        PROJECT_ID, LOCATION, PROCESSOR_ID_LAYOUT, content, mimetype
    )

    if not doc.pages or len(doc.pages) == 0:
        print("[WARNING] No pages detected using Layout. Retrying with OCR processor...")
        doc = process_document_with_processor(
            PROJECT_ID, LOCATION, PROCESSOR_ID_OCR, content, mimetype
        )

    print(f"[DEBUG] Parsed {len(doc.pages)} pages")
    for i, page in enumerate(doc.pages):
        print(f"[DEBUG] Page {i}: {len(page.blocks)} blocks, {len(page.paragraphs)} paragraphs, {len(page.tables)} tables")

    page_tables = tables_for_doc(doc)
    ordered_boxes = reconstruct_boxes(doc)

    plain, html_doc = render_markdown_html(ordered_boxes, page_tables)

    # (Optional) skip redaction for debugging
    # plain = regex_redact(plain)
    # html_doc = regex_redact(html_doc)

    try:
        assert_clean(plain)
    except Exception as e:
        print(f"[WARNING] assert_clean failed: {e}")

    struct = {
        "source": source_id,
        "pages": len(doc.pages),
        "boxes": [vars(b) for b in ordered_boxes],
        "tables": page_tables
    }

    base = pathlib.Path(source_id).stem
    out_txt  = f"{OUTPUT_BUCKET.rstrip('/')}/{base}_document_full.txt"
    out_json = f"{OUTPUT_BUCKET.rstrip('/')}/{base}_document_structured.json"
    out_html = f"{OUTPUT_BUCKET.rstrip('/')}/{base}_document_preview.html"

    write_gcs_bytes(out_txt,  plain.encode("utf-8"), "text/plain; charset=utf-8")
    write_gcs_bytes(out_json, json.dumps(struct, ensure_ascii=False, indent=2).encode("utf-8"), "application/json")
    write_gcs_bytes(out_html, html_doc.encode("utf-8"), "text/html; charset=utf-8")

    print("Wrote:")
    print(" ", out_txt)
    print(" ", out_json)
    print(" ", out_html)


if __name__ == "__main__":
    main()
