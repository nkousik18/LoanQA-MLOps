"""
bias_analysis.py
----------------
Performs bias detection (data slicing) on normalized Textract JSONs.
Generates:
  - bias_report.json (structured report)
  - bias_summary.yaml (readable summary)
  - bias_chart.png (visualization)
  - bias_mitigation_notes.txt (plain-text explanation)
  - fairlearn_bias_metrics.yaml (optional Fairlearn metrics)

Logic:
  - Uses filename + text to infer document category
  - Skips dummy/test/sample/trial files
  - Focuses on categories like 'agreement' and 'application'
"""

import os, sys, json, statistics, yaml
from collections import defaultdict
import matplotlib.pyplot as plt
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
from scripts.aws_extraction_scripts.config import NORMALIZED_DIR, REPORTS_ROOT
from scripts.aws_extraction_scripts.log_utils import get_logger

logger = get_logger("bias_analysis")

# ---------------------------------------------------------------------
# 🧠 Categorization logic
# ---------------------------------------------------------------------
def infer_category(filename: str, text_content: str = "") -> str:
    """
    Categorize file based on filename + text context.
    Priority:
      1. 'agreement' or 'contract' → 'agreement'
      2. Loan/application terms → 'application'
      3. Otherwise → 'other'
    """
    name = filename.lower()
    text = text_content.lower()
    combined = name + " " + text

    if "agreement" in name or "contract" in name:
        return "agreement"

    if "agreement" in text[: len(text) // 6] or "contract" in text[: len(text) // 6]:
        return "agreement"

    application_terms = [
        "loan", "form", "application", "apply", "request",
        "credit", "borrower", "details", "personal information"
    ]
    loan_count = sum(combined.count(term) for term in application_terms)
    if loan_count >= 2:
        return "application"

    return "other"

# ---------------------------------------------------------------------
# 🧩 Analyze bias + generate reports
# ---------------------------------------------------------------------
def analyze_bias():
    normalized_dir = NORMALIZED_DIR
    reports_dir = REPORTS_ROOT
    reports_dir.mkdir(parents=True, exist_ok=True)

    results = defaultdict(list)
    file_counts = defaultdict(int)

    logger.info(f"📁 Scanning normalized files in: {normalized_dir}")

    for file_path in normalized_dir.glob("*_normalized.json"):
        filename = file_path.name

        # 🚫 Skip dummy/test/sample/trial files
        if any(kw in filename.lower() for kw in ["dummy", "test", "sample", "trial"]):
            logger.info(f"🧪 Skipping test file: {filename}")
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"⚠️ Skipping {filename}: {e}")
            continue

        if not isinstance(data, list) or not data:
            logger.info(f"🚫 Skipping empty or invalid file: {filename}")
            continue

        confidences = [entry.get("conf", 0) for entry in data if "conf" in entry]
        if not confidences:
            continue

        if max(confidences) <= 1.0:
            confidences = [round(c * 100, 2) for c in confidences]

        text_content = " ".join(str(d.get("text", "")) for d in data if d.get("text"))
        category = infer_category(filename, text_content)
        results[category].extend(confidences)
        file_counts[category] += 1

        logger.info(f"✅ Processed {filename} ({len(confidences)} spans) → {category}")

    if not results:
        logger.warning("⚠️ No valid normalized files found for bias analysis.")
        return {}

    # ---------------- Compute summary statistics ----------------
    summary = {}
    for cat, values in results.items():
        summary[cat] = {
            "file_count": file_counts[cat],
            "avg_confidence": round(statistics.mean(values), 2),
            "total_spans": len(values),
            "min_confidence": round(min(values), 2),
            "max_confidence": round(max(values), 2),
        }

    valid_cats = [c for c in summary.keys() if c not in ["other", "bias_overview"]]
    if len(valid_cats) >= 2:
        min_cat = min(valid_cats, key=lambda c: summary[c]["avg_confidence"])
        max_cat = max(valid_cats, key=lambda c: summary[c]["avg_confidence"])
        bias_gap = summary[max_cat]["avg_confidence"] - summary[min_cat]["avg_confidence"]
        bias_detected = bias_gap > 5
    else:
        min_cat, max_cat, bias_gap, bias_detected = "N/A", "N/A", 0, False

    summary["bias_overview"] = {
        "lowest_conf_category": min_cat,
        "highest_conf_category": max_cat,
        "confidence_gap": round(bias_gap, 2),
        "bias_detected": bias_detected,
    }

    # ---------------- Save core reports ----------------
    json_path = reports_dir / "bias_report.json"
    yaml_path = reports_dir / "bias_summary.yaml"
    chart_path = reports_dir / "bias_chart.png"
    notes_path = reports_dir / "bias_mitigation_notes.txt"
    fairlearn_path = reports_dir / "fairlearn_bias_metrics.yaml"

    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(summary, jf, indent=2)
    with open(yaml_path, "w", encoding="utf-8") as yf:
        yaml.dump(summary, yf, sort_keys=False)

    # ---------------- 🎨 Visualization ----------------
    plt.figure(figsize=(8, 5))
    cats = [c for c in summary.keys() if c != "bias_overview"]
    avgs = [summary[c]["avg_confidence"] for c in cats]

    min_cat = summary["bias_overview"]["lowest_conf_category"]
    max_cat = summary["bias_overview"]["highest_conf_category"]

    colors = []
    for c in cats:
        if c == min_cat:
            colors.append("salmon")
        elif c == max_cat:
            colors.append("mediumseagreen")
        else:
            colors.append("cornflowerblue")

    bars = plt.bar(cats, avgs, color=colors, edgecolor="black")
    for bar, val in zip(bars, avgs):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            val + 1,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.axhline(statistics.mean(avgs), color="red", linestyle="--", label="Global Mean")
    plt.title("OCR Confidence by Document Category (Bias Visualization)")
    plt.xlabel("Document Category")
    plt.ylabel("Average Confidence (%)")
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()

    # ---------------- 🧾 Mitigation Notes ----------------
    with open(notes_path, "w", encoding="utf-8") as f:
        f.write("Bias Mitigation Notes\n=====================\n\n")
        if bias_detected:
            f.write(
                f"⚠️ Bias Detected: '{min_cat}' category has "
                f"{round(bias_gap, 2)}% lower confidence than '{max_cat}'.\n\n"
            )
            f.write("Recommended Mitigation Steps:\n")
            f.write("1. Increase samples of low-confidence document type.\n")
            f.write("2. Check OCR quality and layout consistency.\n")
            f.write("3. Apply data augmentation or fine-tuning for fairness.\n")
        else:
            f.write("✅ No significant bias detected among document categories.\n")

    # ---------------- ⚖️ Fairlearn Integration ----------------
    try:
        from fairlearn.metrics import MetricFrame, selection_rate
        from sklearn.metrics import accuracy_score
        import numpy as np

        threshold = 90
        fairness_data = []
        for cat, confs in results.items():
            y_true = np.ones(len(confs))
            y_pred = np.array([1 if c >= threshold else 0 for c in confs])
            fairness_data.append((cat, y_true, y_pred))

        metrics = {}
        for cat, y_true, y_pred in fairness_data:
            frame = MetricFrame(
                metrics={"accuracy": accuracy_score, "selection_rate": selection_rate},
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=[cat] * len(y_true)
            )
            metrics[cat] = {
                "accuracy": float(frame.overall["accuracy"]),
                "selection_rate": float(frame.overall["selection_rate"])
            }

        with open(fairlearn_path, "w", encoding="utf-8") as f:
            yaml.dump(metrics, f, sort_keys=False)
        logger.info("✅ Fairlearn bias metrics saved.")
    except Exception as e:
        logger.warning(f"⚠️ Fairlearn integration skipped: {e}")

    # ---------------- ✅ Summary ----------------
    logger.info(f"✅ Bias analysis completed. Reports saved to {reports_dir}")
    print("\n🎯 Bias Analysis Completed Successfully!")
    print("📊 Generated files:")
    print(" - bias_report.json")
    print(" - bias_summary.yaml")
    print(" - bias_chart.png")
    print(" - bias_mitigation_notes.txt")
    print(" - fairlearn_bias_metrics.yaml\n")

    return summary

# ---------------------------------------------------------------------
# 🏁 Entry Point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    analyze_bias()
