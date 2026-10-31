
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"
TRAIN_PATH = DATA_DIR / "Training.csv"
TEST_PATH = DATA_DIR / "Testing.csv"
SYNTH_PATH = DATA_DIR / "synthetic" / "synthetic_candidates.csv"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(exist_ok=True)

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
synthetic = pd.read_csv(SYNTH_PATH)

def clean_columns(df):
    df.columns = df.columns.astype(str).str.strip()
    df.drop(
        columns=[c for c in df.columns if c.startswith("Unnamed")],
        inplace=True,
        errors="ignore"
    )
    return df

train = clean_columns(train)
test = clean_columns(test)
synthetic = clean_columns(synthetic)

for name, df in [
    ("Training", train),
    ("Testing", test),
    ("Synthetic", synthetic)
]:
    if "prognosis" not in df.columns:
        raise SystemExit(f"{name}.csv has no prognosis column.")

# Use symptom features present in all three datasets.
features = [
    c for c in train.columns
    if c != "prognosis"
    and c in test.columns
    and c in synthetic.columns
]

if not features:
    raise SystemExit("No common symptom columns were found.")

print("Training symptom columns:", len(train.columns) - 1)
print("Testing symptom columns:", len(test.columns) - 1)
print("Synthetic symptom columns:", len(synthetic.columns) - 1)
print("Common symptom columns used:", len(features))

missing_test = sorted(
    set(train.columns) - set(test.columns) - {"prognosis"}
)
if missing_test:
    print("Training symptoms missing from Testing.csv:", missing_test)

missing_synthetic = sorted(
    set(train.columns) - set(synthetic.columns) - {"prognosis"}
)
if missing_synthetic:
    print("Training symptoms missing from synthetic data:", missing_synthetic)

# Verify labels and feature values.
train = train[features + ["prognosis"]].copy()
test = test[features + ["prognosis"]].copy()
synthetic = synthetic[features + ["prognosis"]].copy()

for name, df in [
    ("Training", train),
    ("Testing", test),
    ("Synthetic", synthetic)
]:
    for feature in features:
        if not df[feature].dropna().isin([0, 1]).all():
            raise SystemExit(
                f"{name}.csv has non-binary values in {feature}."
            )

    if df[features].isna().any().any():
        raise SystemExit(f"{name}.csv contains missing symptom values.")

X_test = test[features]
y_test = test["prognosis"]

def evaluate(name, training_data):
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42
    )
    model.fit(training_data[features], training_data["prognosis"])

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    print(f"\n=== {name} ===")
    print("Training rows:", len(training_data))
    print("Test rows:", len(y_test))
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    return {
        "Experiment": name,
        "Training_Rows": len(training_data),
        "Test_Rows": len(y_test),
        "Features_Used": len(features),
        "Accuracy": accuracy,
        "Macro_F1": macro_f1
    }

original_result = evaluate("Original data only", train)

augmented_train = pd.concat(
    [train, synthetic],
    ignore_index=True
)
augmented_result = evaluate(
    "Original + synthetic candidates",
    augmented_train
)

results = pd.DataFrame([original_result, augmented_result])
output_path = REPORT_DIR / "synthetic_comparison.csv"
results.to_csv(output_path, index=False)

print("\n=== Comparison ===")
print(results.to_string(index=False))
print("\nSaved to:", output_path)
print("\nEducational experiment only; not clinical validation.")
