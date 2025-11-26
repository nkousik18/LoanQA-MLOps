import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")      # <--- Fix for macOS no-display issue

import matplotlib.pyplot as plt
import seaborn as sns

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
PLOT_DIR = os.path.join(ROOT, "evaluation_results/model_selection/plots")


def ensure_dir():
    os.makedirs(PLOT_DIR, exist_ok=True)


def load_summary(path=None):
    if path is None:
        path = os.path.join(ROOT, "evaluation_results/prompt_eval/summary.csv")
    return pd.read_csv(path)


def plot_accuracy(df):
    ensure_dir()
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="model", y="correct", errorbar=None)
    plt.title("Grounded Answer Accuracy by Model")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=20)
    plt.savefig(os.path.join(PLOT_DIR, "accuracy_by_model.png"), dpi=300)
    plt.close()


def plot_hallucination(df):
    ensure_dir()
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="model", y="hallucinated", errorbar=None)
    plt.title("Hallucination Rate by Model")
    plt.ylabel("Hallucination Rate")
    plt.xticks(rotation=20)
    plt.savefig(os.path.join(PLOT_DIR, "hallucination_by_model.png"), dpi=300)
    plt.close()


def plot_latency(df):
    ensure_dir()
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="model", y="llm_latency", errorbar=None)
    plt.title("Avg LLM Latency by Model")
    plt.ylabel("Latency (seconds)")
    plt.xticks(rotation=20)
    plt.savefig(os.path.join(PLOT_DIR, "latency_by_model.png"), dpi=300)
    plt.close()


def generate_all_plots():
    df = load_summary()
    df["correct"] = df["verdict"].apply(lambda x: 1 if x == "grounded" else 0)

    plot_accuracy(df)
    plot_hallucination(df)
    plot_latency(df)

    print("Saved plots to:", PLOT_DIR)


if __name__ == "__main__":
    generate_all_plots()
