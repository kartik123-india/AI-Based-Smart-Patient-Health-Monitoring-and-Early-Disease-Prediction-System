
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

BASE = Path(__file__).resolve().parent
DATA = BASE / "dataset"

train = pd.read_csv(DATA / "Training.csv")
test = pd.read_csv(DATA / "Testing.csv")
synthetic = pd.read_csv(
    DATA / "synthetic" / "synthetic_candidates.csv"
)

for df in (train, test, synthetic):
    df.columns = df.columns.astype(str).str.strip()
    df.drop(
        columns=[c for c in df.columns if c.startswith("Unnamed")],
        inplace=True,
        errors="ignore"
    )

features = [c for c in train.columns if c != "prognosis"]

if not (
    set(test.columns) == set(train.columns)
    and set(synthetic.columns) == set(train.columns)
):
    raise SystemExit("Dataset columns do not match.")

test = test[features + ["prognosis"]]
synthetic = synthetic[features + ["prognosis"]]

X_test = test[features]
y_test = test["prognosis"]

baseline = RandomForestClassifier(
    n_estimators=300, random_state=42
)
baseline.fit(train[features], train["prognosis"])
baseline_pred = baseline.predict(X_test)

augmented_train = pd.concat(
    [train, synthetic], ignore_index=True
)

augmented = RandomForestClassifier(
    n_estimators=300, random_state=42
)
augmented.fit(
    augmented_train[features],
    augmented_train["prognosis"]
)
augmented_pred = augmented.predict(X_test)

results = pd.DataFrame({
    "Actual": y_test,
    "Original_Prediction": baseline_pred,
    "Augmented_Prediction": augmented_pred
})

results["Original_Correct"] = (
    results["Actual"] == results["Original_Prediction"]
)
results["Augmented_Correct"] = (
    results["Actual"] == results["Augmented_Prediction"]
)

print("\n=== Test cases corrected by augmentation ===")
print(
    results[
        (~results["Original_Correct"])
        & results["Augmented_Correct"]
    ].to_string(index=True)
)

print("\n=== Test cases made incorrect by augmentation ===")
print(
    results[
        results["Original_Correct"]
        & (~results["Augmented_Correct"])
    ].to_string(index=True)
)

print("\n=== All test predictions ===")
print(results.to_string(index=True))

output = BASE / "reports" / "synthetic_prediction_inspection.csv"
results.to_csv(output, index=True, index_label="Test_Row")
print("\nSaved to:", output)
