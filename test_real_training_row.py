
from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "dataset" / "Training.csv")
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# Find a real Common Cold example containing both symptoms
rows = df[
    (df["prognosis"] == "Common Cold") &
    (df["high_fever"] == 1) &
    (df["runny_nose"] == 1)
]

if rows.empty:
    print("No matching training row found.")
    raise SystemExit

features = [c for c in df.columns if c != "prognosis"]
X = rows[features].iloc[[0]]

print("Actual training label:", rows.iloc[0]["prognosis"])
print("Symptoms present:", [
    col for col in features if X.iloc[0][col] == 1
])

for filename in ["disease_model.joblib", "disease_model_augmented.joblib"]:
    obj = joblib.load(BASE_DIR / "models" / filename)

    if isinstance(obj, dict):
        model = obj["model"]
        model_features = obj["feature_columns"]
    else:
        model = obj
        model_features = list(model.feature_names_in_)

    prediction = model.predict(X[model_features])[0]
    print(f"\n{filename}")
    print("Prediction:", prediction)
    print("Correct:", prediction == rows.iloc[0]["prognosis"])

print("\nTest complete.")
