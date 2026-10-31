from pathlib import Path
import warnings

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "dataset" / "Training.csv"
TEST_PATH = BASE_DIR / "dataset" / "Testing.csv"
MODEL_PATH = BASE_DIR / "models" / "disease_model_augmented.joblib"

warnings.filterwarnings("ignore")

def main():
    print("=" * 65)
    print("AI HEALTH PROJECT - MODEL RELIABILITY CHECK")
    print("=" * 65)

    for path in [TRAIN_PATH, TEST_PATH, MODEL_PATH]:
        if not path.exists():
            print(f"\nMISSING FILE: {path}")
            return

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    model = joblib.load(MODEL_PATH)

    if isinstance(model, dict):
        print("\nThe augmented model file contains a dictionary wrapper.")
        if "model" not in model:
            print("Cannot find a 'model' entry in the dictionary.")
            return
        model = model["model"]

    features = list(getattr(model, "feature_names_in_", []))
    if not features:
        features = [c for c in train_df.columns if c != "prognosis"]

    missing_train = [c for c in features if c not in train_df.columns]
    missing_test = [c for c in features if c not in test_df.columns]
    if missing_train or missing_test:
        print("\nFeature mismatch detected.")
        print("Missing from Training.csv:", missing_train[:20])
        print("Missing from Testing.csv:", missing_test[:20])
        return

    print("\n1. DATASET OVERVIEW")
    print("Training rows:", len(train_df))
    print("Testing rows:", len(test_df))
    print("Model features:", len(features))
    print("Model classes:", len(getattr(model, "classes_", [])))

    if "prognosis" in train_df.columns:
        print("Training disease classes:", train_df["prognosis"].nunique())
        print("\nTraining examples per disease (lowest counts):")
        print(train_df["prognosis"].astype(str).str.strip().value_counts().tail(10).to_string())

    X_test = test_df[features].copy()
    predictions = model.predict(X_test)

    print("\n2. TEST DATA PREDICTION DISTRIBUTION")
    print(pd.Series(predictions).astype(str).str.strip().value_counts().to_string())

    if "prognosis" in test_df.columns:
        y_test = test_df["prognosis"].astype(str).str.strip()
        pred_clean = pd.Series(predictions).astype(str).str.strip()
        print("\n3. HELD-OUT TEST METRICS")
        print("Accuracy:", round(accuracy_score(y_test, pred_clean), 4))
        print("\nClassification report:")
        print(classification_report(y_test, pred_clean, zero_division=0))
        labels = sorted(set(y_test) | set(pred_clean))
        cm = confusion_matrix(y_test, pred_clean, labels=labels)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        print("\nConfusion matrix saved to model_confusion_matrix.csv")
        cm_df.to_csv(BASE_DIR / "model_confusion_matrix.csv")
    else:
        print("\nTesting.csv has no 'prognosis' column.")
        print("A true accuracy score cannot be calculated from this file.")

    print("\n4. AIDS-RELATED DATA CHECK")
    if "prognosis" in train_df.columns:
        aids_rows = train_df[
            train_df["prognosis"].astype(str).str.strip().str.lower() == "aids"
        ]
        print("AIDS training rows:", len(aids_rows))
        if len(aids_rows):
            active_counts = aids_rows[features].sum(axis=1)
            print("Average active symptoms per AIDS row:", round(active_counts.mean(), 2))
            print("Most frequent symptoms in AIDS training rows:")
            print(
                aids_rows[features].sum().sort_values(ascending=False)
                .head(20).to_string()
            )

    print("\n5. YOUR EXAMPLE INPUT")
    example_symptoms = ["high_fever", "vomiting"]
    valid = [s for s in example_symptoms if s in features]
    absent = [s for s in example_symptoms if s not in features]
    print("Requested symptoms:", example_symptoms)
    print("Matched feature labels:", valid)
    print("Not present as exact feature labels:", absent)

    if valid:
        row = {feature: 0 for feature in features}
        for symptom in valid:
            row[symptom] = 1
        example_df = pd.DataFrame([row], columns=features)
        print("Model output for this exact two-symptom input:",
              str(model.predict(example_df)[0]).strip())

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(example_df)[0]
            top_indices = probabilities.argsort()[::-1][:5]
            print("Top model scores (not calibrated medical probabilities):")
            for index in top_indices:
                print(
                    f"  {str(model.classes_[index]).strip()}: "
                    f"{probabilities[index] * 100:.2f}%"
                )

    print("\nIMPORTANT:")
    print("- This is a technical dataset/model audit, not clinical validation.")
    print("- Model scores are not medical probabilities.")
    print("- Do not use these outputs to diagnose or treat anyone.")
    print("\nCheck completed.")

if __name__ == "__main__":
    main()
