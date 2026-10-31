
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "dataset" / "Training.csv"
SYNTH_PATH = (
    BASE_DIR / "dataset" / "synthetic"
    / "synthetic_candidates.csv"
)

train = pd.read_csv(TRAIN_PATH)
synthetic = pd.read_csv(SYNTH_PATH)

train.columns = train.columns.astype(str).str.strip()
synthetic.columns = synthetic.columns.astype(str).str.strip()

train = train.loc[:, ~train.columns.str.startswith("Unnamed")]
synthetic = synthetic.loc[:, ~synthetic.columns.str.startswith("Unnamed")]

features = [c for c in train.columns if c != "prognosis"]

print("\n=== Synthetic Dataset Validation ===")
print("Original training rows:", len(train))
print("Synthetic candidate rows:", len(synthetic))
print("Original disease classes:", train["prognosis"].nunique())
print("Synthetic disease classes:", synthetic["prognosis"].nunique())

missing_columns = set(train.columns) - set(synthetic.columns)
extra_columns = set(synthetic.columns) - set(train.columns)

print("\nMissing columns:", sorted(missing_columns))
print("Extra columns:", sorted(extra_columns))

if missing_columns or extra_columns:
    print("\nColumn mismatch: do not combine these datasets yet.")
    raise SystemExit(1)

valid_labels = set(train["prognosis"].dropna().astype(str))
synthetic_labels = set(synthetic["prognosis"].dropna().astype(str))

unknown_labels = synthetic_labels - valid_labels
print("Unknown disease labels:", sorted(unknown_labels))

invalid_values = {}
for feature in features:
    values = set(synthetic[feature].dropna().unique())
    invalid = values - {0, 1, 0.0, 1.0}
    if invalid:
        invalid_values[feature] = list(invalid)

print("Features with non-binary values:", invalid_values)

missing_values = int(synthetic.isna().sum().sum())
print("Total missing cells:", missing_values)

print("\nCandidate records per class:")
print(synthetic["prognosis"].value_counts().to_string())

print("\nExact duplicate rows:", int(synthetic.duplicated().sum()))

# Compare each synthetic feature vector to original training vectors.
original_vectors = set(
    train[features].astype(int).astype(str).agg(",".join, axis=1)
)
synthetic_vectors = (
    synthetic[features].astype(int).astype(str).agg(",".join, axis=1)
)
overlap = int(synthetic_vectors.isin(original_vectors).sum())

print("Synthetic rows matching an original symptom vector:", overlap)

print("\nValidation here checks structure only.")
print("It does not establish medical correctness.")

print("\n=== Additional Checks ===")

print("\nNumber of symptoms per synthetic row:")
print(synthetic[features].sum(axis=1).describe())

print("\nRows with fewer than 2 active symptoms:")
print(
    synthetic.loc[
        synthetic[features].sum(axis=1) < 2,
        ["prognosis"]
    ]
)

print("\nOverlapping synthetic records:")
overlap_mask = synthetic_vectors.isin(original_vectors)
print(synthetic.loc[overlap_mask, ["prognosis"]])
