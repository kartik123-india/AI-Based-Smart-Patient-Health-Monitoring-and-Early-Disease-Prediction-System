
from pathlib import Path
import pandas as pd

base = Path(__file__).resolve().parent

train_path = base / "dataset" / "Training.csv"
synthetic_path = (
    base / "dataset" / "synthetic" / "synthetic_candidates.csv"
)

train = pd.read_csv(train_path)
synthetic = pd.read_csv(synthetic_path)

for df in (train, synthetic):
    df.columns = df.columns.astype(str).str.strip()
    df.drop(
        columns=[c for c in df.columns if c.startswith("Unnamed:")],
        inplace=True,
        errors="ignore"
    )

features = [c for c in train.columns if c != "prognosis"]

original_labels = {}

for _, row in train.iterrows():
    pattern = tuple(row[features].fillna(0))
    original_labels.setdefault(pattern, set()).add(row["prognosis"])

keep = []

for _, row in synthetic.iterrows():
    pattern = tuple(row[features].fillna(0))
    labels = original_labels.get(pattern, set())

    conflict = bool(labels) and row["prognosis"] not in labels
    keep.append(not conflict)

removed = len(keep) - sum(keep)
cleaned = synthetic.loc[keep].copy()

backup_path = synthetic_path.with_name(
    "synthetic_candidates_backup.csv"
)
if not backup_path.exists():
    synthetic.to_csv(backup_path, index=False)

cleaned.to_csv(synthetic_path, index=False)

print("Conflicting records removed:", removed)
print("Remaining synthetic records:", len(cleaned))
print("Backup saved to:", backup_path)
print("Updated file:", synthetic_path)
