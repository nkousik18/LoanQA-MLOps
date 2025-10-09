import pandas as pd

# Paths
finrad_path = "/Users/kousiknandury/PycharmProjects/DL_project/data/Finance_terms_definitions_labels.csv"
output_path = "/Users/kousiknandury/PycharmProjects/DL_project/data/financial_terms.csv"

# Load dataset
df = pd.read_csv(finrad_path)
print(f"[INFO] Loaded dataset with {len(df)} rows")

# Inspect columns
print(f"[INFO] Columns found: {list(df.columns)}")

# ✅ Keep only the first two columns (usually term + definition)
# Rename them to standard names for your pipeline
df = df.iloc[:, :2]
df.columns = ["terms", "definitions"]

# Drop duplicates or NaNs
df = df.dropna(subset=["terms"]).drop_duplicates(subset=["terms"])

# Clean whitespace
df["terms"] = df["terms"].str.strip()
df["definitions"] = df["definitions"].astype(str).str.strip()

# Save clean glossary
df.to_csv(output_path, index=False)
print(f"[INFO] Saved glossary with {len(df)} unique terms → {output_path}")
