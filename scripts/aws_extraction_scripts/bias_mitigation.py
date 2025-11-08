"""
bias_mitigation.py
------------------
Stage 7: Generates post-bias analysis recommendations.

Purpose:
  - Reads bias_report.json (from bias_analysis.py)
  - Identifies the lowest-performing document category
  - Produces actionable, human-readable mitigation recommendations
  - Saves results as reports/aws_extraction_reports/mitigation_manifest.yaml

Notes:
  - Does NOT modify or duplicate data
  - Provides guidance for dataset balancing and retraining fairness
"""

import os, sys, json, yaml
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------
# 🔧 Ensure project root and imports
# ---------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# ---------------------------------------------------------------------
# 📦 Imports from centralized config and logger
# ---------------------------------------------------------------------
from scripts.aws_extraction_scripts.config import REPORTS_ROOT
from scripts.aws_extraction_scripts.log_utils import get_logger

logger = get_logger("bias_mitigation")

# ---------------------------------------------------------------------
# 🧩 Main Function
# ---------------------------------------------------------------------
def recommend_bias_mitigation():
    """
    Reads bias_report.json → Generates YAML recommendations for bias mitigation.
    """
    reports_dir = REPORTS_ROOT
    bias_report_path = reports_dir / "bias_report.json"
    manifest_path = reports_dir / "mitigation_manifest.yaml"

    if not bias_report_path.exists():
        logger.error("❌ Missing bias_report.json. Run bias_analysis.py first.")
        print("\n⚠️ Cannot generate mitigation recommendations: bias_report.json not found.\n")
        return

    # ---------------------------------------------------------------
    # 1️⃣ Load bias report
    # ---------------------------------------------------------------
    with open(bias_report_path, "r", encoding="utf-8") as f:
        bias_data = json.load(f)

    bias_info = bias_data.get("bias_overview", {})
    low_cat = bias_info.get("lowest_conf_category", "")
    gap = bias_info.get("confidence_gap", 0)
    bias_detected = bias_info.get("bias_detected", False)

    # ---------------------------------------------------------------
    # 2️⃣ Generate recommendations
    # ---------------------------------------------------------------
    if not bias_detected or not low_cat or low_cat == "N/A":
        logger.info("✅ No significant bias detected. No mitigation needed.")
        mitigation_notes = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "No bias detected",
            "message": "Confidence levels across categories are within acceptable fairness thresholds.",
            "next_steps": [
                "Continue collecting balanced datasets across document types.",
                "Re-run bias_analysis.py periodically after adding new data."
            ]
        }
    else:
        logger.info(f"⚙️ Bias detected in category: {low_cat} (gap ≈ {gap}%)")

        mitigation_notes = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "low_performing_category": low_cat,
            "confidence_gap_percent": round(gap, 2),
            "analysis_summary": (
                f"The '{low_cat}' category shows lower OCR confidence "
                f"compared to other document types."
            ),
            "recommended_real_data_actions": [
                f"Collect additional real-world '{low_cat}' documents (+25–30% more samples).",
                f"Re-run OCR normalization and recheck average confidence.",
                f"Perform targeted data augmentation (e.g., rotation, lighting normalization).",
                f"Verify template consistency (headers, signature blocks, layout structure).",
                f"If bias persists, fine-tune OCR or apply domain-specific preprocessing."
            ],
            "fairlearn_or_modeling_actions": [
                "Use Fairlearn’s EqualizedOdds or DemographicParity post-processing.",
                "Re-evaluate confidence parity using the fairness dashboard.",
                "Apply sample reweighting for low-confidence categories during retraining."
            ],
            "notes": (
                "This diagnostic suggests mitigation strategies only — "
                "it does not alter or duplicate data."
            )
        }

    # ---------------------------------------------------------------
    # 3️⃣ Save YAML manifest
    # ---------------------------------------------------------------
    with open(manifest_path, "w", encoding="utf-8") as f:
        yaml.dump(mitigation_notes, f, sort_keys=False)

    logger.info(f"🧾 Mitigation recommendations saved → {manifest_path}")

    print("\n✅ Bias Mitigation Recommendation Completed!")
    print(f"📊 Results saved to: {manifest_path}\n")

    return mitigation_notes

# ---------------------------------------------------------------------
# 🏁 Entry Point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    recommend_bias_mitigation()
