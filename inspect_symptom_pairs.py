
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "dataset" / "Training.csv")

# Remove unnecessary unnamed columns
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

tests = {
    "Fever + runny nose": ["high_fever", "runny_nose"],
    "Itching + skin rash": ["itching", "skin_rash"],
    "Headache + fatigue": ["headache", "fatigue"],
}

print("Dataset shape:", df.shape)
print("Label column: prognosis")
print()

for name, symptoms in tests.items():
    print("=" * 50)
    print(name)

    missing = [s for s in symptoms if s not in df.columns]
    if missing:
        print("Missing features:", missing)
        continue

    # Rows containing both symptoms
    mask = (df[symptoms] == 1).all(axis=1)
    matched = df.loc[mask]

    print("Training rows containing BOTH symptoms:", len(matched))

    if len(matched):
        print("\nDisease counts for these exact symptom pairs:")
        print(matched["prognosis"].value_counts().to_string())

    print("\nDisease counts for each symptom separately:")
    for symptom in symptoms:
        print(f"\n{symptom}:")
        rows = df.loc[df[symptom] == 1]
        print(rows["prognosis"].value_counts().head(10).to_string())

print("\nInspection complete.")
