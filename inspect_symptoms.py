
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "Training.csv"

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

symptoms_to_check = [
    "high_fever",
    "runny_nose",
    "headache",
    "fatigue",
    "itching",
    "skin_rash",
]

print("Dataset shape:", df.shape)
print("\nDisease labels:")
print(df["prognosis"].value_counts().to_string())

for symptom in symptoms_to_check:
    if symptom not in df.columns:
        print(f"\n{symptom}: NOT FOUND")
        continue

    print(f"\n{symptom}:")
    present = df[df[symptom] == 1]

    print("Records with symptom:", len(present))
    print("Most common labels when present:")
    print(present["prognosis"].value_counts().head(10).to_string())
