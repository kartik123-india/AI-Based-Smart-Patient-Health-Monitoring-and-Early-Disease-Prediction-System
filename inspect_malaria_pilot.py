
import json
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent

FILE = (
    BASE / "dataset" / "synthetic" /
    "malaria_pilot.csv"
)

df = pd.read_csv(FILE)

print("=== MALARIA PILOT REVIEW ===")
print("Total records:", len(df))

for index, row in df.iterrows():

    symptoms = json.loads(row["symptoms"])

    print(f"\nRecord {index + 1}")
    print("Disease:", row["prognosis"])
    print("Symptoms:", ", ".join(symptoms))
    print("Status:", row["validation_status"])

print("\n=== SUMMARY ===")
print("Unique combinations:", df["symptoms"].nunique())
print("Disease labels:", df["prognosis"].nunique())

print("\nReview completed.")
