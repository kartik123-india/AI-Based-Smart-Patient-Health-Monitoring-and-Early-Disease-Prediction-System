
from pathlib import Path
import pandas as pd

base = Path(__file__).resolve().parent

train = pd.read_csv(base / "dataset" / "Training.csv")
synthetic = pd.read_csv(
    base / "dataset" / "synthetic" / "synthetic_candidates.csv"
)

for df in (train, synthetic):
    df.columns = df.columns.astype(str).str.strip()
    df.drop(
        columns=[c for c in df.columns if c.startswith("Unnamed:")],
        inplace=True,
        errors="ignore"
    )

features = [c for c in train.columns if c != "prognosis"]

original = train[features + ["prognosis"]].copy()
original[features] = original[features].fillna(0)

synthetic[features] = synthetic[features].fillna(0)

original_groups = {}

for _, row in original.iterrows():
    pattern = tuple(row[features])
    original_groups.setdefault(pattern, set()).add(row["prognosis"])

results = []

for _, row in synthetic.iterrows():
    pattern = tuple(row[features])

    if pattern in original_groups:
        labels = sorted(original_groups[pattern])
        results.append({
            "Synthetic_Label": row["prognosis"],
            "Original_Label(s)": ", ".join(labels),
            "Labels_Match": row["prognosis"] in labels
        })

result = pd.DataFrame(results)

print("\n=== Overlap Label Check ===")
print(result.to_string(index=False))

if not result.empty:
    print("\nTotal overlaps:", len(result))
    print("Matching labels:", result["Labels_Match"].sum())
    print("Conflicting labels:", (~result["Labels_Match"]).sum())

    output = base / "reports" / "overlap_label_check.csv"
    result.to_csv(output, index=False)
    print("\nSaved to:", output)
