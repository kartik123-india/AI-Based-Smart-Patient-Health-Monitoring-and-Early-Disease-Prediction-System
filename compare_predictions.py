
from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

model_paths = {
    "Original model": BASE_DIR / "models" / "disease_model.joblib",
    "Augmented model": BASE_DIR / "models" / "disease_model_augmented.joblib",
}

tests = {
    "Fever + runny nose": ["high_fever", "runny_nose"],
    "Itching + skin rash": ["itching", "skin_rash"],
    "Headache + fatigue": ["headache", "fatigue"],
}

for name, path in model_paths.items():
    print(f"\n{'=' * 45}\n{name}")

    saved_object = joblib.load(path)

    if isinstance(saved_object, dict):
        model = saved_object["model"]
        features = saved_object["feature_columns"]
    else:
        model = saved_object
        features = list(model.feature_names_in_)

    print("Model type:", type(model).__name__)

    for test_name, symptoms in tests.items():
        row = {feature: 0 for feature in features}

        for symptom in symptoms:
            if symptom in row:
                row[symptom] = 1
            else:
                print(f"WARNING: Feature not found: {symptom}")

        X = pd.DataFrame([row], columns=features)
        prediction = model.predict(X)[0]

        print(f"\n{test_name}: {prediction}")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)[0]
            top_indices = probabilities.argsort()[-3:][::-1]

            print("Top 3 model scores:")
            for index in top_indices:
                print(
                    f"  {model.classes_[index]}: "
                    f"{probabilities[index]:.3f}"
                )
