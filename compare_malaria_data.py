
import json
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent

TRAINING_FILE = BASE / "dataset" / "Training.csv"

SYNTHETIC_FILE = (
    BASE / "dataset" / "synthetic" /
    "malaria_pilot.csv"
)

OUTPUT_FILE = (
    BASE / "reports" /
    "malaria_data_comparison.csv"
)

def main():

    original = pd.read_csv(TRAINING_FILE)
    original["prognosis"] = (
        original["prognosis"].astype(str).str.strip()
    )

    original = original[
        original["prognosis"] == "Malaria"
    ]

    synthetic = pd.read_csv(SYNTHETIC_FILE)

    original_features = original.drop(
        columns=["prognosis", "Unnamed: 133"],
        errors="ignore"
    )

    original_counts = original_features.sum()

    synthetic_counts = pd.Series(
        [
            symptom
            for row in synthetic["symptoms"]
            for symptom in json.loads(row)
        ]
    ).value_counts()

    symptoms = sorted(
        set(original_counts.index) |
        set(synthetic_counts.index)
    )

    report = pd.DataFrame({
        "symptom": symptoms,
        "original_count": [
            int(original_counts.get(s, 0))
            for s in symptoms
        ],
        "synthetic_count": [
            int(synthetic_counts.get(s, 0))
            for s in symptoms
        ]
    })

    report["original_percentage"] = (
        report["original_count"] / len(original) * 100
    ).round(2)

    report["synthetic_percentage"] = (
        report["synthetic_count"] / len(synthetic) * 100
    ).round(2)

    report["frequency_difference"] = (
        report["synthetic_percentage"] -
        report["original_percentage"]
    ).round(2)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    report.to_csv(OUTPUT_FILE, index=False)

    print("=== MALARIA DATA COMPARISON ===")
    print(report.to_string(index=False))

    print("\nOriginal records:", len(original))
    print("Synthetic records:", len(synthetic))
    print("Report saved:", OUTPUT_FILE)

    print("\nNote: Frequencies are descriptive,")
    print("not clinical prevalence estimates.")

if __name__ == "__main__":
    main()
