from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "Training.csv"

def main():
    if not DATA_PATH.exists():
        print("Training.csv not found:", DATA_PATH)
        return

    df = pd.read_csv(DATA_PATH)
    required = ["high_fever", "vomiting", "prognosis"]
    missing = [column for column in required if column not in df.columns]

    if missing:
        print("Required columns missing:", missing)
        print("Available columns include:", list(df.columns[:15]))
        return

    # Treat nonzero values as symptom present.
    high_fever = pd.to_numeric(df["high_fever"], errors="coerce").fillna(0) == 1
    vomiting = pd.to_numeric(df["vomiting"], errors="coerce").fillna(0) == 1

    df = df.copy()
    df["high_fever_present"] = high_fever
    df["vomiting_present"] = vomiting

    both = df[high_fever & vomiting]
    fever_only = df[high_fever & ~vomiting]
    vomiting_only = df[~high_fever & vomiting]
    neither = df[~high_fever & ~vomiting]

    print("=" * 68)
    print("SYMPTOM CO-OCCURRENCE AUDIT: HIGH FEVER + VOMITING")
    print("=" * 68)
    print("Total training records:", len(df))
    print("\nSymptom combination counts:")
    print("Both high fever and vomiting:", len(both))
    print("High fever only:", len(fever_only))
    print("Vomiting only:", len(vomiting_only))
    print("Neither symptom:", len(neither))

    print("\nDisease distribution when BOTH symptoms are present:")
    if both.empty:
        print("No records contain both symptoms.")
    else:
        counts = both["prognosis"].astype(str).str.strip().value_counts()
        print(counts.to_string())
        counts.rename_axis("Disease").reset_index(name="Training records").to_csv(
            BASE_DIR / "high_fever_vomiting_disease_counts.csv", index=False
        )
        print("\nSaved: high_fever_vomiting_disease_counts.csv")

    print("\nDisease distribution for HIGH FEVER only:")
    if fever_only.empty:
        print("No records contain high fever without vomiting.")
    else:
        print(
            fever_only["prognosis"].astype(str).str.strip()
            .value_counts().head(15).to_string()
        )

    print("\nDisease distribution for VOMITING only:")
    if vomiting_only.empty:
        print("No records contain vomiting without high fever.")
    else:
        print(
            vomiting_only["prognosis"].astype(str).str.strip()
            .value_counts().head(15).to_string()
        )

    print("\nAIDS-specific check:")
    aids = df[df["prognosis"].astype(str).str.strip().str.lower() == "aids"]
    print("AIDS rows:", len(aids))
    print("AIDS rows with high fever:", int(aids["high_fever_present"].sum()))
    print("AIDS rows with vomiting:", int(aids["vomiting_present"].sum()))
    print("AIDS rows with BOTH:", int(
        (aids["high_fever_present"] & aids["vomiting_present"]).sum()
    ))

    print("\nNote: These are dataset pattern counts, not medical evidence.")
    print("Do not use them to diagnose or treat anyone.")

if __name__ == "__main__":
    main()
