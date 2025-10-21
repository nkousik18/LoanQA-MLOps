import os

# === Auto-resolve project root (DL_project) ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# === Dynamic data and output directories ===
DATA_DIR = os.path.join(BASE_DIR, "data", "loan_docs")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "clean_texts")

# === Make sure output folder exists ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Config constants ===
MIN_TEXT_LEN = 40
LANG = "en"

print(f"[CONFIG] Base Directory: {BASE_DIR}")
print(f"[CONFIG] Data Directory: {DATA_DIR}")
print(f"[CONFIG] Output Directory: {OUTPUT_DIR}")
