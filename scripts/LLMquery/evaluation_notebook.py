"""
evaluation_notebook.py
Compute routing accuracy, hallucination rate, latency stats
"""

import pandas as pd, numpy as np, matplotlib.pyplot as plt

df = pd.read_csv("logs/query_logs.csv")

print(f"🧾 Loaded {len(df)} log entries")

# ---- Basic stats ----
print("Average time:", df["time_taken_sec"].mean())
print("Average confidence:", df["confidence"].mean())

# ---- Routing precision ----
if "expected_intent" in df.columns:
    routing_acc = (df["intent"] == df["expected_intent"]).mean()
    print(f"Routing Accuracy: {routing_acc*100:.1f}%")

# ---- Hallucination proxy ----
hallucinated = df[df["llm_response"].str.contains("specify this information", case=False, na=False)]
print(f"Potential hallucination rate: {len(hallucinated)/len(df)*100:.1f}%")

# ---- Latency histogram ----
plt.hist(df["time_taken_sec"], bins=10, color='steelblue', alpha=0.7)
plt.xlabel("Latency (s)")
plt.ylabel("Count")
plt.title("Response Time Distribution")
plt.show()

# ---- Confidence vs Intent ----
df.groupby("intent")["confidence"].mean().plot(kind="bar", color="orange")
plt.title("Mean Confidence per Intent")
plt.ylabel("Confidence")
plt.show()
