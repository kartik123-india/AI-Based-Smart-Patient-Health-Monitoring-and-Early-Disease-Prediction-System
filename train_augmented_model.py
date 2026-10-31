
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

train_path = Path("dataset/Training.csv")
synthetic_path = Path("dataset/synthetic/synthetic_candidates.csv")
test_path = Path("dataset/Testing.csv")
model_path = Path("models/disease_model_augmented.joblib")

# Load datasets
original = pd.read_csv(train_path)
synthetic = pd.read_csv(synthetic_path)
test = pd.read_csv(test_path)

# Remove unwanted CSV index/unnamed columns
def clean_data(df):
    return df.loc[:, ~df.columns.str.startswith("Unnamed:")].copy()

original = clean_data(original)
synthetic = clean_data(synthetic)
test = clean_data(test)

# Use exactly the same feature columns everywhere
features = [c for c in original.columns if c != "prognosis"]

missing_synthetic = set(features) - set(synthetic.columns)
missing_test = set(features) - set(test.columns)

if missing_synthetic or missing_test:
    raise ValueError(
        f"Missing synthetic columns: {missing_synthetic}\n"
        f"Missing test columns: {missing_test}"
    )

X_train = pd.concat(
    [original[features], synthetic[features]],
    ignore_index=True
)
y_train = pd.concat(
    [original["prognosis"], synthetic["prognosis"]],
    ignore_index=True
)

X_test = test[features]
y_test = test["prognosis"]

# Train and evaluate
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\n=== Augmented Model Results ===")
print("Original training rows:", len(original))
print("Synthetic rows:", len(synthetic))
print("Combined training rows:", len(X_train))
print("Test rows:", len(test))
print("Features:", len(features))
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print("Macro F1:", round(
    f1_score(y_test, predictions, average="macro"), 4
))

# Save separately from the original model
joblib.dump(model, model_path)
print("\nAugmented model saved to:", model_path)
