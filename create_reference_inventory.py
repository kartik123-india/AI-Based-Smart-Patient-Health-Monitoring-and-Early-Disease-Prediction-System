from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent

TRAINING_FILE = BASE / "dataset" / "Training.csv"
OUTPUT_FILE = (
    BASE / "medical_references" /
    "reference_inventory.csv"
)

def main():
    if not TRAINING_FILE.exists():
        print("ERROR: Training.csv not found.")
        return

    if OUTPUT_FILE.exists():
        print("ERROR: Reference inventory already exists.")
        print("Existing file was not overwritten.")
        return

    df = pd.read_csv(TRAINING_FILE)

    if "prognosis" not in df.columns:
        print("ERROR: 'prognosis' column not found.")
        print("Available columns:", list(df.columns))
        return

    diseases = sorted(
        df["prognosis"].dropna().astype(str).unique()
    )

    inventory = pd.DataFrame({
        "disease": diseases,
        "reference_source": "",
        "reference_url": "",
        "reference_status": "NOT_STARTED",
        "review_notes": ""
    })

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    inventory.to_csv(OUTPUT_FILE, index=False)

    print("=== MEDICAL REFERENCE INVENTORY ===")
    print("Total diseases:", len(inventory))
    print("Output:", OUTPUT_FILE)

    print("\nDisease list:")
    for number, disease in enumerate(diseases, 1):
        print(f"{number}. {disease}")

    print("\nInventory created successfully.")

if __name__ == "__main__":
    main()