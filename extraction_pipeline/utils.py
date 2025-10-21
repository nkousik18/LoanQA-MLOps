import os
from datetime import datetime

def save_text(output_text, output_dir, source_file):
    base = os.path.splitext(os.path.basename(source_file))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"{base}_{timestamp}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output_text)
    print(f"✅ Saved extracted text → {out_path}")

def list_files(directory):
    """List all supported files (PDF + images)."""
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith((".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"))
    ]
