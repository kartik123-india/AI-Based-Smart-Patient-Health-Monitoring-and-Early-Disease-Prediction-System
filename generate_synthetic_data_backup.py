
import os
import json
import time
from pathlib import Path

import pandas as pd
from google import genai

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = DATA_DIR / "synthetic"

TRAIN_PATH = DATA_DIR / "Training.csv"
OUTPUT_PATH = OUTPUT_DIR / "synthetic_candidates.csv"

MODEL = "gemini-3.5-flash-lite"
RECORDS_PER_CLASS = 3
BATCH_SIZE = 3

# Wait between requests to reduce rate-limit errors.
REQUEST_DELAY_SECONDS = 5
MAX_RETRIES = 4

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise SystemExit("GEMINI_API_KEY is not set in CMD.")

if not TRAIN_PATH.exists():
    raise FileNotFoundError("dataset/Training.csv was not found.")

df = pd.read_csv(TRAIN_PATH)
df.columns = df.columns.astype(str).str.strip()
df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
df = df.loc[:, ~df.columns.duplicated()]

if "prognosis" not in df.columns:
    raise ValueError("The prognosis column is missing.")

features = [c for c in df.columns if c != "prognosis"]
classes = sorted(df["prognosis"].dropna().astype(str).unique())

# Load existing candidates instead of overwriting them.
if OUTPUT_PATH.exists():
    existing = pd.read_csv(OUTPUT_PATH)
    existing.columns = existing.columns.astype(str).str.strip()

    if set(existing.columns) != set(features + ["prognosis"]):
        raise SystemExit(
            "Existing CSV columns do not match Training.csv. "
            "Back up the file and inspect it before continuing."
        )

    existing = existing[features + ["prognosis"]]
    existing = existing.drop_duplicates()
else:
    existing = pd.DataFrame(columns=features + ["prognosis"])

client = genai.Client(api_key=api_key)


def generate_batch(disease):
    prompt = f"""
Create {BATCH_SIZE} synthetic symptom records for an academic
machine-learning experiment.

Target label: {disease}

Return JSON only:
{{
  "records": [
    {{"symptoms": ["symptom_name", "symptom_name"]}}
  ]
}}

Rules:
- Use only exact symptom names from the approved feature list.
- Each record must contain 2 to 8 distinct symptoms.
- Do not invent symptom names.
- Do not claim these are real patient records.
- Synthetic associations may be medically inaccurate.
- These records must never be used for clinical decisions.

Approved features:
{json.dumps(features)}
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config={"response_mime_type": "application/json"}
    )

    data = json.loads(response.text or "{}")
    records = data.get("records", [])

    if not isinstance(records, list):
        return []

    valid_rows = []

    for item in records:
        if not isinstance(item, dict):
            continue

        symptoms = item.get("symptoms", [])

        if not isinstance(symptoms, list):
            continue

        if not all(
            isinstance(s, str) and s in features
            for s in symptoms
        ):
            continue

        symptoms = set(symptoms)

        if not 2 <= len(symptoms) <= 8:
            continue

        row = {f: int(f in symptoms) for f in features}
        row["prognosis"] = disease
        valid_rows.append(row)

    return valid_rows


for disease in classes:
    current_count = int(
        (existing["prognosis"].astype(str) == disease).sum()
    )

    if current_count >= RECORDS_PER_CLASS:
        continue

    print(f"\n{disease}: {current_count}/{RECORDS_PER_CLASS} candidates")

    while current_count < RECORDS_PER_CLASS:
        rows = []
        succeeded = False

        for attempt in range(MAX_RETRIES):
            try:
                rows = generate_batch(disease)
                succeeded = True
                break

            except Exception as error:
                message = str(error).lower()
                is_rate_limit = (
                    "429" in message
                    or "resource_exhausted" in message
                    or "quota" in message
                )

                if is_rate_limit:
                    wait = 30 * (attempt + 1)
                    print(
                        f"Rate limit reached. Waiting {wait} seconds "
                        f"(attempt {attempt + 1}/{MAX_RETRIES})."
                    )
                    time.sleep(wait)
                else:
                    print(
                        f"Request failed: {type(error).__name__}: {error}"
                    )
                    break

        if not succeeded:
            print(f"Pausing generation for {disease}.")
            break

        # Do not add a symptom vector already present in the candidates
        # or in the original training data.
        original_vectors = set(
            df[features].astype(int).astype(str).agg(",".join, axis=1)
        )
        candidate_vectors = set(
            existing[features].astype(int).astype(str).agg(",".join, axis=1)
        )

        added = 0

        for row in rows:
            vector = ",".join(str(int(row[f])) for f in features)

            if vector in original_vectors or vector in candidate_vectors:
                continue

            existing = pd.concat(
                [existing, pd.DataFrame([row])],
                ignore_index=True
            )
            candidate_vectors.add(vector)
            added += 1

            current_count = int(
                (existing["prognosis"].astype(str) == disease).sum()
            )

            if current_count >= RECORDS_PER_CLASS:
                break

        # Save progress after each response, including when no row was added.
        existing = existing.drop_duplicates()
        existing.to_csv(OUTPUT_PATH, index=False)

        print(f"Added {added} new candidate(s).")
        print(f"Progress saved. {current_count}/{RECORDS_PER_CLASS} for {disease}.")

        if added == 0:
            print("No new unique symptom patterns; moving to the next class.")
            break

        if current_count < RECORDS_PER_CLASS:
            time.sleep(REQUEST_DELAY_SECONDS)

print("\nGeneration attempt completed.")
print(f"Total candidate rows: {len(existing)}")
print(f"Disease classes represented: {existing['prognosis'].nunique()}")
print(f"Saved to: {OUTPUT_PATH}")
print("Candidates are unverified and not medically validated.")
