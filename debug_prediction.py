
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "disease_model_augmented.joblib"

model = joblib.load(MODEL_PATH)
features = list(model.feature_names_in_)

tests = {
    "Fever and runny nose": ["high_fever", "runny_nose"],
    "Itching and skin rash": ["itching", "skin_rash"],
    "Headache and fatigue": ["headache", "fatigue"],
}

for test_name, symptoms in tests.items():
    row = {feature: 0 for feature in features}

    for symptom in symptoms:
        if symptom in row:
            row[symptom] = 1
        else:
            print(f"Feature not found: {symptom}")

    X = pd.DataFrame([row], columns=features)
    prediction = model.predict(X)[0]

    print(f"\nTest: {test_name}")
    print(f"Predicted class: {prediction}")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]
        top_indices = probabilities.argsort()[-3:][::-1]

        print("Top 3 model scores:")
        for index in top_indices:
            print(
                f"  {model.classes_[index]}: "
                f"{probabilities[index]:.3f}"
            )
