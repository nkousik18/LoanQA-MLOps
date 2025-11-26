"""
generate_schema_stats.py
------------------------
Stage 5: Generates schema & validation statistics from normalized OCR JSON files.

Inputs:
  data/aws_extraction_data/normalized/*.json  -> normalized OCR outputs

Outputs (in data/aws_extraction_data/schema/):
  - expectations.json      -> inferred schema + validation rules
  - validation_result.json -> validation statistics & rule outcomes

Features:
- Centralized config-driven paths
- Structured logging (local + Airflow)
- Robust schema inference & validation
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any
import pandas as pd

# ---------------------------------------------------------------------
# 🔧 Ensure working directory and imports
# ---------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# ---------------------------------------------------------------------
# 📦 Imports from centralized config and logger
# ---------------------------------------------------------------------
from scripts.aws_extraction_scripts.config import NORMALIZED_DIR, SCHEMA_DIR
from scripts.aws_extraction_scripts.log_utils import get_logger
from scripts.aws_extraction_scripts.tracker import track_task

logger = get_logger("schema_stats")

# ---------------------------------------------------------------------
# 📥 Load normalized JSON files
# ---------------------------------------------------------------------
def load_normalized_data() -> pd.DataFrame:
    """Loads all normalized JSONs into a single DataFrame."""
    rows = []
    files = list(NORMALIZED_DIR.glob("*.json"))

    if not files:
        logger.warning(f"⚠️ No normalized JSON files found in {NORMALIZED_DIR}")
        return pd.DataFrame()

    for f in files:
        try:
            with f.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, list):
                    rows.extend(data)
                else:
                    logger.warning(f"{f.name} did not contain a list at top level; skipping.")
        except Exception as e:
            logger.error(f"Error reading {f.name}: {e}")

    df = pd.DataFrame(rows)
    logger.info(f"✅ Loaded {len(df)} rows from {len(files)} normalized files.")
    return df

# ---------------------------------------------------------------------
# 🧠 Infer schema and statistics
# ---------------------------------------------------------------------
def infer_schema_and_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Infers column schema and computes summary statistics."""
    info: Dict[str, Any] = {"columns": {}, "row_count": int(len(df))}

    for col in df.columns:
        s = df[col]
        non_null, nulls = s.notna().sum(), s.isna().sum()
        dtype_str = str(s.dtype)

        try:
            sample_val = s.dropna().iloc[0]
            hash(sample_val)
            can_hash = True
        except Exception:
            can_hash = False

        if can_hash:
            unique = int(s.nunique(dropna=True))
            example_values = s.dropna().unique()[:5].tolist()
        else:
            s_str = s.astype(str)
            unique = int(s_str.nunique(dropna=True))
            example_values = s_str.dropna().unique()[:5].tolist()

        col_info = {
            "dtype": dtype_str,
            "non_null_count": int(non_null),
            "null_count": int(nulls),
            "null_ratio": float(nulls / len(df)) if len(df) else 0.0,
            "unique_values": unique,
            "example_values": example_values,
        }

        if pd.api.types.is_numeric_dtype(s):
            col_info.update({
                "min": float(s.min()) if non_null else None,
                "max": float(s.max()) if non_null else None,
                "mean": float(s.mean()) if non_null else None,
            })

        info["columns"][col] = col_info

    return info

# ---------------------------------------------------------------------
# 🧾 Data validation rules
# ---------------------------------------------------------------------
def validate_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Applies lightweight data quality rules."""
    results: Dict[str, Any] = {"total_rows": int(len(df)), "rules": {}, "overall_passed": True}

    def add_rule(name: str, passed: bool, details: Dict[str, Any] = None):
        results["rules"][name] = {"passed": bool(passed), "details": details or {}}
        if passed:
            logger.info(f"✅ Validation rule passed: {name}")
        else:
            logger.warning(f"⚠️ Validation rule FAILED: {name} | {details}")
            results["overall_passed"] = False

    # Required columns
    required_cols = ["doc_id", "page", "text", "conf"]
    missing = [c for c in required_cols if c not in df.columns]
    add_rule("required_columns_present", len(missing) == 0, {"required": required_cols, "missing": missing})

    if "text" in df.columns:
        null_text = int(df["text"].isna().sum())
        add_rule("text_not_null", null_text == 0, {"null_text_count": null_text})

    if "conf" in df.columns:
        conf = df["conf"]
        invalid_low = int((conf < 0).sum())
        invalid_high = int((conf > 100).sum())
        add_rule("conf_in_range_0_100", invalid_low == 0 and invalid_high == 0,
                 {"below_0": invalid_low, "above_100": invalid_high})

    return results

# ---------------------------------------------------------------------
# 🧮 Main: Generate schema + validation stats
# ---------------------------------------------------------------------
def generate_schema_stats(**context):
    """
    Main entry to generate expectations + validation results.
    Returns paths to schema files (string) for Airflow XCom.
    """
    task_name = "generate_schema_stats"
    track_task(task_name, "STARTED")

    try:
        logger.info("🚀 Starting schema & validation statistics generation...")

        df = load_normalized_data()
        if df.empty:
            msg = "❌ No normalized data found. Exiting schema stats generation."
            logger.error(msg)
            track_task(task_name, "FAILED", error=msg)
            return None

        SCHEMA_DIR.mkdir(parents=True, exist_ok=True)

        schema_stats = infer_schema_and_stats(df)
        validation_results = validate_data(df)

        expectations = {
            "description": "Inferred schema and validation rules for normalized OCR output.",
            "schema": schema_stats,
            "rules": validation_results["rules"],
        }

        expectations_path = SCHEMA_DIR / "expectations.json"
        validation_path = SCHEMA_DIR / "validation_result.json"

        with expectations_path.open("w", encoding="utf-8") as f:
            json.dump(expectations, f, indent=2)

        with validation_path.open("w", encoding="utf-8") as f:
            json.dump(validation_results, f, indent=2)

        logger.info(f"✅ Schema & validation results saved: {expectations_path}, {validation_path}")
        track_task(task_name, "SUCCESS", details="Schema and validation generated")

        print("\n✅ Schema & statistics generated successfully!")
        print(f"📘 Schema & rules:   {expectations_path}")
        print(f"📊 Validation stats: {validation_path}")

        # ✅ Return for Airflow XCom
        return [str(expectations_path), str(validation_path)]

    except Exception as e:
        err = f"❌ Schema generation failed: {e}"
        logger.exception(err)
        track_task(task_name, "FAILED", error=str(e))
        return None


# ---------------------------------------------------------------------
# 🏁 Entry Point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    generate_schema_stats()
