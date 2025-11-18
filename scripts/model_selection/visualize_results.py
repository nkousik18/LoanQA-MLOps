# scripts/model_selection/visualize_results.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


PLOT_DIR = "evaluation_results/model_selection/plots/"


def ensure_dir():
    os.makedirs(PLOT_DIR, exist_ok=True)


def load_summary(path="evaluation_results/model_selection/summary.csv"):
    return pd.read_csv(path)


# ---------------------------------------------------------
# Plot: Accuracy by Model
# ---------------------------------------------------------
def plot_accuracy(df):
    ensure_dir()
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df,
        x="model",
        y="correct",
        estimator="mean",
        ci=None
    )
    plt.title("Grounded Answer Accuracy by Model")
    plt.ylabel("Accuracy (mean correctness)")
    plt.xticks(rotation=20)
    plt.savefig(PLOT_DIR + "accuracy_by_model.png", dpi=300)
    plt.close()


# ---------------------------------------------------------
# Plot: Hallucination Rate
# ---------------------------------------------------------
def plot_hallucination(df):
    ensure_dir()
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df,
        x="model",
        y="hallucinated",
        estimator="mean",
        ci=None
    )
    plt.title("Hallucination Rate by Model")
    plt.ylabel("Hallucination Rate")
    plt.xticks(rotation=20)
    plt.savefig(PLOT_DIR + "hallucination_by_model.png", dpi=300)
    plt.close()


# ---------------------------------------------------------
# Plot: LLM Latency
# ---------------------------------------------------------
def plot_latency(df):
    ensure_dir()
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df,
        x="model",
        y="llm_latency",
        estimator="mean",
        ci=None
    )
    plt.title("Average LLM Latency by Model")
    plt.ylabel("Latency (seconds)")
    plt.xticks(rotation=20)
    plt.savefig(PLOT_DIR + "latency_by_model.png", dpi=300)
    plt.close()


# ---------------------------------------------------------
# Plot: Template Comparison
# ---------------------------------------------------------
def plot_templates(df):
    ensure_dir()
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df,
        x="template",
        y="correct",
        estimator="mean",
        hue="model",
        ci=None
    )
    plt.title("Template Effect on Accuracy Across Models")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=20)
    plt.savefig(PLOT_DIR + "template_accuracy_comparison.png", dpi=300)
    plt.close()


# ---------------------------------------------------------
# Plot: JSON Validity (for JSON template)
# ---------------------------------------------------------
def plot_json_validity(df):
    df_json = df[df["template"] == "json"]
    if df_json.empty:
        return

    ensure_dir()
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df_json,
        x="model",
        y="json_valid",
        estimator="mean",
        ci=None
    )
    plt.title("JSON Valid Output Rate (JSON Template)")
    plt.ylabel("Valid JSON Rate")
    plt.xticks(rotation=20)
    plt.savefig(PLOT_DIR + "json_validity_by_model.png", dpi=300)
    plt.close()


# ---------------------------------------------------------
# Master function to generate all plots
# ---------------------------------------------------------
def generate_all_plots():
    df = load_summary()
    plot_accuracy(df)
    plot_hallucination(df)
    plot_latency(df)
    plot_templates(df)
    plot_json_validity(df)
    print("All plots saved to:", PLOT_DIR)


if __name__ == "__main__":
    generate_all_plots()
