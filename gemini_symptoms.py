
import os
import json
from pathlib import Path

import joblib
from google import genai

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "disease_model.joblib"

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise SystemExit(
        "GEMINI_API_KEY is missing. Set it in CMD first."
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Trained model not found. Run python train_model.py first."
    )

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
features = artifact["feature_columns"]
allowed_features = set(features)

client = genai.Client(api_key=api_key)

def interpret_symptoms(description):
    prompt = f"""
You are a symptom-text extraction component for an academic
health project. You do not diagnose diseases.

Convert the user's description into dataset symptom names.

Rules:
- Return JSON only, with the key "symptoms".
- "symptoms" must be a list of exact names from the approved list.
- Include only symptoms the user explicitly reports.
- Do not infer symptoms or diseases.
- If no listed symptom matches, return an empty list.
- Do not provide medical advice or a diagnosis.

Approved symptom names:
{json.dumps(features)}

User description:
{description}

Required JSON format:
{{"symptoms": ["itching", "skin_rash"]}}
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
        config={
            "response_mime_type": "application/json"
        }
    )

    data = json.loads(response.text)
    symptoms = data.get("symptoms", [])

    if not isinstance(symptoms, list):
        raise ValueError("Gemini returned an invalid symptom list.")

    # Enforce the approved feature list in local Python code.
    return sorted({
        symptom for symptom in symptoms
        if isinstance(symptom, str)
        and symptom in allowed_features
    })


print("\n=== AI Health Project: Gemini + ML ===")
print("Academic prototype; not a medical diagnosis.\n")

description = input(
    "Describe your symptoms in English or Hinglish: "
).strip()

if not description:
    raise SystemExit("No description entered.")

try:
    symptoms = interpret_symptoms(description)

    if not symptoms:
        print("\nNo recognized dataset symptoms found.")
        print("Try describing your symptoms more clearly.")
    else:
        print("\nRecognized symptom features:")
        for symptom in symptoms:
            print("-", symptom)

        # This is a simplified prototype:
        # unmentioned symptoms are treated as absent.
        row = {feature: 0 for feature in features}

        for symptom in symptoms:
            row[symptom] = 1

        import pandas as pd
        input_df = pd.DataFrame([row], columns=features)
        prediction = model.predict(input_df)[0]

        print("\nModel-predicted class:", prediction)
        print("\nThis result is not a medical diagnosis.")
        print("Consult a qualified healthcare professional.")

except Exception as error:
    print("\nCould not process the description.")
    print(type(error).__name__, error)
