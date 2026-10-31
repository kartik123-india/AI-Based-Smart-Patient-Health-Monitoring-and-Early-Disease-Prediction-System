
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

TRAIN_PATH = DATA_DIR / "Training.csv"
TEST_PATH = DATA_DIR / "Testing.csv"


def load_dataset(path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    df.columns = df.columns.astype(str).str.strip()
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df = df.loc[:, ~df.columns.duplicated()]

    if "prognosis" not in df.columns:
        raise ValueError(
            f"'prognosis' column missing in {path.name}"
        )

    df = df.dropna(subset=["prognosis"])
    df["prognosis"] = df["prognosis"].astype(str).str.strip()
    return df


def make_models():
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced"
        ),
        "Naive Bayes": BernoulliNB(),
        "SVM": Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("scaler", StandardScaler()),
            ("model", SVC())
        ]),
        "KNN": Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5))
        ]),
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000))
        ])
    }


print("Loading datasets...")

train_df = load_dataset(TRAIN_PATH)
test_df = load_dataset(TEST_PATH)

features = [
    col for col in train_df.columns if col != "prognosis"
]

if not features:
    raise ValueError("No symptom columns found.")

missing = set(features) - set(test_df.columns)
if missing:
    raise ValueError(
        f"Testing.csv is missing columns: {sorted(missing)}"
    )

X = train_df[features].apply(pd.to_numeric, errors="coerce")
y = train_df["prognosis"]

X_test = test_df[features].apply(pd.to_numeric, errors="coerce")
y_test = test_df["prognosis"]

if X.isna().all(axis=1).any():
    raise ValueError("A training row contains no numeric features.")

if y.nunique() < 2:
    raise ValueError("At least two disease classes are required.")

# Split training data for fair model comparison.
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

results = []
models = make_models()

for name, model in models.items():
    print(f"\nTraining {name}...")

    fitted = clone(model)
    fitted.fit(X_train, y_train)
    predictions = fitted.predict(X_val)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_val, predictions),
        "Macro_F1": f1_score(
            y_val, predictions,
            average="macro",
            zero_division=0
        )
    })

results_df = pd.DataFrame(results).sort_values(
    ["Macro_F1", "Accuracy"],
    ascending=False
)

print("\n=== Model Comparison ===")
print(results_df.to_string(index=False))

results_df.to_csv(
    REPORT_DIR / "model_comparison.csv",
    index=False
)

best_name = results_df.iloc[0]["Model"]
best_model = clone(models[best_name])

print(f"\nBest model: {best_name}")
print("Training the selected model on all training rows...")

best_model.fit(X, y)

# Evaluate the selected model on the provided test dataset.
known_labels = set(y)
mask = y_test.isin(known_labels)

if mask.any():
    test_predictions = best_model.predict(X_test.loc[mask])

    test_accuracy = accuracy_score(
        y_test.loc[mask], test_predictions
    )
    test_f1 = f1_score(
        y_test.loc[mask],
        test_predictions,
        average="macro",
        zero_division=0
    )
else:
    test_accuracy = None
    test_f1 = None

artifact = {
    "model": best_model,
    "feature_columns": features,
    "model_name": best_name,
    "classes": sorted(y.unique().tolist()),
    "dataset_note": "Educational dataset; not clinically validated."
}

joblib.dump(artifact, MODEL_DIR / "disease_model.joblib")

summary = {
    "selected_model": best_name,
    "training_rows": len(X),
    "symptom_features": len(features),
    "disease_classes": int(y.nunique()),
    "validation_accuracy": float(
        results_df.iloc[0]["Accuracy"]
    ),
    "validation_macro_f1": float(
        results_df.iloc[0]["Macro_F1"]
    ),
    "external_test_rows_evaluated": int(mask.sum()),
    "external_test_accuracy": test_accuracy,
    "external_test_macro_f1": test_f1
}

with open(
    REPORT_DIR / "training_summary.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(summary, file, indent=2)

print("\n=== Provided Testing.csv Results ===")
print(f"Rows evaluated: {int(mask.sum())}")
print(f"Test accuracy: {test_accuracy}")
print(f"Test macro F1: {test_f1}")

print("\nModel saved: models\\disease_model.joblib")
print("Comparison saved: reports\\model_comparison.csv")
print("Summary saved: reports\\training_summary.json")
print("\nTraining completed successfully.")
