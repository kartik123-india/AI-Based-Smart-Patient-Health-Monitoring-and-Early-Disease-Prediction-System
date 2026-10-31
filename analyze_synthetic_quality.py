
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
train = pd.read_csv(BASE / "dataset" / "Training.csv")
synthetic = pd.read_csv(
    BASE / "dataset" / "synthetic" / "synthetic_candidates.csv"
)

def clean(df):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    return df

train = clean(train)
synthetic = clean(synthetic)

features = [c for c in train.columns if c != "prognosis"]
synthetic = synthetic[features + ["prognosis"]]

train_patterns = set(
    train[features].astype(int).astype(str).agg(",".join, axis=1)
)
synthetic_patterns = synthetic[features].astype(int).astype(str).agg(
    ",".join, axis=1
)

print("=== Synthetic pattern analysis ===")
print("Synthetic rows:", len(synthetic))
print("Unique synthetic symptom patterns:", synthetic_patterns.nunique())
print(
    "Patterns already present in training:",
    synthetic_patterns.isin(train_patterns).sum()
)
print(
    "New symptom patterns:",
    (~synthetic_patterns.isin(train_patterns)).sum()
)

print("\n=== Disease label counts ===")
print(synthetic["prognosis"].value_counts().sort_index().to_string())

print("\n=== Label distribution in original training data ===")
for _, row in synthetic.iterrows():
    matches = train[
        (train[features].astype(int) == row[features].astype(int)).all(axis=1)
    ]
    if not matches.empty:
        print(
            "Exact match:",
            row["prognosis"],
            "| Original label(s):",
            ", ".join(sorted(matches["prognosis"].unique()))
        )

print("\nAnalysis complete.")
