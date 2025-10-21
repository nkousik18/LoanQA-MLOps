# layout_aware_do_not_use/utils/io_utils.py
from pathlib import Path

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
