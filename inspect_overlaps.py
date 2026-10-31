
from pathlib import Path
import pandas as pd

base = Path(__file__).resolve().parent

train = pd.read_csv(base / "dataset" / "Training.csv")
synthetic = pd.read_csv(
    base / "dataset" / "synthetic" / "synthetic_candidates.csv"
)

# Remove accidental index columns and clean column names
for df in (train, synthetic):
    df.columns = df.columns.astype(str).str.strip()
    df.drop(
        columns=[c for c in df.columns if c.startswith("Unnamed:")],
        inplace=True,
        errors="ignore"
    )

features = [c for c in train.columns if c != "prognosis"]

missing = set(features + ["prognosis"]) - set(synthetic.columns)
if missing:
    raise SystemExit(f"Missing columns: {sorted(missing)}")

original_patterns = {
    tuple(row)
    for row in train[features].fillna(0).to_numpy()
}

synthetic_features = synthetic[features].fillna(0)

mask = synthetic_features.apply(
    lambda row: tuple(row) in original_patterns, axis=1
)

overlaps = synthetic.loc[mask, ["prognosis"] + features]

print("Overlapping rows:", len(overlaps))
print(overlaps[["prognosis"] + features].to_string(index=False))

output = base / "reports" / "overlapping_synthetic_rows.csv"
overlaps.to_csv(output, index=False)
print("\nSaved to:", output)
