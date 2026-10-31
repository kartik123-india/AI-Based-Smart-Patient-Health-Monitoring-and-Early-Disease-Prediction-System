import json
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE / "dataset" / "synthetic" /
    "common_cold_pilot.csv"
)

OUTPUT_FILE = (
    BASE / "dataset" / "synthetic" /
    "common_cold_pilot_reference_filtered.csv"
)

ALLOWED_SYMPTOMS = {
    "continuous_sneezing",
    "runny_nose",
    "congestion",
    "cough",
    "throat_irritation",
    "headache"
}

def main():
    if not INPUT_FILE.exists():
        print("ERROR: Pilot CSV not found.")
        return

    df = pd.read_csv(INPUT_FILE)

    accepted = []
    rejected = []

    for index, row in df.iterrows():
        try:
            symptoms = json.loads(row["symptoms"])

            if not isinstance(symptoms, list):
                raise ValueError("Symptoms must be a list.")

            if (
                len(symptoms) >= 2
                and len(symptoms) == len(set(symptoms))
                and set(symptoms).issubset(ALLOWED_SYMPTOMS)
            ):
                accepted.append(row)
            else:
                rejected.append({
                    "record": index + 1,
                    "symptoms": symptoms
                })

        except Exception as error:
            rejected.append({
                "record": index + 1,
                "error": str(error)
            })

    if accepted:
        result = pd.DataFrame(accepted)
        result = result.drop_duplicates(
            subset=["prognosis", "symptoms"]
        )
    else:
        result = pd.DataFrame(columns=df.columns)

    result.to_csv(OUTPUT_FILE, index=False)

    print("\n=== FILTERING REPORT ===")
    print("Original records:", len(df))
    print("Accepted records:", len(result))
    print("Rejected records:", len(rejected))

    print("\n=== REJECTED RECORDS ===")
    for item in rejected:
        print(item)

    print("\nOutput file:")
    print(OUTPUT_FILE)

    print("\nIMPORTANT:")
    print("Filtering checks symptom names and structure only.")
    print("It does not medically validate the records.")
    print("Expert review is still required before training.")

if __name__ == "__main__":
    main()