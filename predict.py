
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "disease_model_augmented.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError("Augmented model missing. Run python train_augmented_model.py first.")

model = joblib.load(MODEL_PATH)

# The augmented model was saved directly, not as a dictionary.
features = list(model.feature_names_in_)

print("\n=== AI Health Project ===")
print("Educational prototype, not a medical diagnosis.\n")

print("Enter the exact symptom names, separated by commas.")
print("Example: itching, skin_rash")

entered = input("\nEnter symptoms: ").strip().lower()

selected = {
    symptom.strip()
    for symptom in entered.split(",")
    if symptom.strip()
}

unknown = selected - set(features)

if unknown:
    print("\nUnrecognized symptoms:")
    print(", ".join(sorted(unknown)))
    print("Please use the exact symptom names.")
elif not selected:
    print("\nPlease enter at least one symptom.")
else:
    row = {feature: 0 for feature in features}

    for symptom in selected:
        row[symptom] = 1

    input_df = pd.DataFrame([row], columns=features)
    prediction = model.predict(input_df)[0]

    print("\nModel-predicted class:", prediction)
    print("\nThis is an educational prediction, not a diagnosis.")
    print("Consult a qualified healthcare professional.")
