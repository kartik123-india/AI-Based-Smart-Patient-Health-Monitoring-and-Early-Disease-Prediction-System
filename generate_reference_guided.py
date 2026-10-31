
import os
import json
import re
from pathlib import Path

import pandas as pd
from google import genai


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

BASE = Path(__file__).resolve().parent

REFERENCE_FILE = (
    BASE / "medical_references" / "diabetes.json"
)

MODEL = "gemini-3.5-flash-lite"

DEFAULT_BATCH_SIZE = 10
DEFAULT_MIN_SYMPTOMS = 2
DEFAULT_MAX_SYMPTOMS = 8


# ==========================================
# LOAD AND VALIDATE REFERENCE
# ==========================================

def load_reference():

    if not REFERENCE_FILE.exists():
        raise FileNotFoundError(
            f"Reference file not found: {REFERENCE_FILE}"
        )

    with open(REFERENCE_FILE, "r", encoding="utf-8") as file:
        reference = json.load(file)

    required = [
        "disease",
        "reference_source",
        "reference_url",
        "symptom_categories"
    ]

    for field in required:
        if field not in reference:
            raise ValueError(
                f"Missing reference field: {field}"
            )

    categories = reference["symptom_categories"]

    common = categories.get(
        "commonly_documented", []
    )

    variable = (
        categories.get("variable_or_less_typical")
        or categories.get("variable_or_type_specific", [])
    )

    if not isinstance(common, list):
        raise ValueError(
            "Invalid common symptom category."
        )

    if not isinstance(variable, list):
        raise ValueError(
            "Invalid variable symptom category."
        )

    if not common:
        raise ValueError(
            "No commonly documented symptoms."
        )

    all_symptoms = common + variable

    if not all(
        isinstance(symptom, str) and symptom.strip()
        for symptom in all_symptoms
    ):
        raise ValueError(
            "Invalid symptom names."
        )

    if len(set(all_symptoms)) != len(all_symptoms):
        raise ValueError(
            "Duplicate symptoms found."
        )

    settings = reference.get(
        "generation_settings", {}
    )

    if not isinstance(settings, dict):
        raise ValueError(
            "generation_settings must be a JSON object."
        )

    batch_size = settings.get(
        "batch_size", DEFAULT_BATCH_SIZE
    )

    min_symptoms = settings.get(
        "min_symptoms", DEFAULT_MIN_SYMPTOMS
    )

    max_symptoms = settings.get(
        "max_symptoms",
        min(DEFAULT_MAX_SYMPTOMS, len(all_symptoms))
    )

    for name, value in [
        ("batch_size", batch_size),
        ("min_symptoms", min_symptoms),
        ("max_symptoms", max_symptoms)
    ]:
        if type(value) is not int or value < 1:
            raise ValueError(
                f"{name} must be a positive integer."
            )

    if min_symptoms > max_symptoms:
        raise ValueError(
            "Minimum symptoms cannot exceed maximum symptoms."
        )

    if max_symptoms > len(all_symptoms):
        raise ValueError(
            "Maximum symptoms exceeds the available reference vocabulary."
        )

    reference["common_symptoms"] = common
    reference["variable_symptoms"] = variable

    reference["batch_size"] = batch_size
    reference["min_symptoms"] = min_symptoms
    reference["max_symptoms"] = max_symptoms

    return reference


# ==========================================
# CREATE DISEASE-SPECIFIC OUTPUT PATH
# ==========================================

def get_output_file(reference):

    disease = reference["disease"].strip()

    disease_slug = re.sub(
        r"[^a-z0-9]+",
        "_",
        disease.lower()
    ).strip("_")

    return (
        BASE
        / "dataset"
        / "synthetic"
        / f"{disease_slug}_pilot.csv"
    )


# ==========================================
# GENERATE RECORDS USING GEMINI
# ==========================================

def generate_records(client, reference):

    prompt = f"""
Generate up to {reference["batch_size"]} synthetic
educational symptom combinations for:

Disease: {reference["disease"]}

Medical reference:
{reference["reference_source"]}

Reference URL:
{reference["reference_url"]}

Commonly documented symptoms:
{json.dumps(reference["common_symptoms"])}

Variable or less typical symptoms:
{json.dumps(reference["variable_symptoms"])}

Required symptom count:
Minimum: {reference["min_symptoms"]}
Maximum: {reference["max_symptoms"]}

Return JSON only in this format:

{{
  "records": [
    {{"symptoms": ["symptom_one", "symptom_two"]}}
  ]
}}

Rules:

1. Use only the supplied symptom names.
2. Prefer commonly documented symptoms.
3. Use variable symptoms only when appropriate.
4. Follow the requested symptom-count range.
5. Generate varied combinations where supported.
6. Do not invent symptoms.
7. Do not invent disease probabilities.
8. Do not claim these are real patient records.
9. Do not claim combinations are medically validated.
10. Do not force medically unsupported combinations.
11. Return fewer records rather than fabricate information.
12. Return JSON only, without explanations.
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config={
            "response_mime_type": "application/json"
        }
    )

    if not response.text:
        raise ValueError(
            "Gemini returned an empty response."
        )

    result = json.loads(response.text)

    records = result.get("records", [])

    if not isinstance(records, list):
        raise ValueError(
            "Invalid Gemini response."
        )

    return records


# ==========================================
# STRUCTURAL VALIDATION
# ==========================================

def validate_records(records, reference):

    disease = reference["disease"]

    common = set(reference["common_symptoms"])
    variable = set(reference["variable_symptoms"])

    allowed = common | variable

    min_symptoms = reference["min_symptoms"]
    max_symptoms = reference["max_symptoms"]

    accepted = []
    rejected = 0

    for record in records:

        if not isinstance(record, dict):
            rejected += 1
            continue

        symptoms = record.get("symptoms")

        if not isinstance(symptoms, list):
            rejected += 1
            continue

        if not all(
            isinstance(symptom, str)
            and symptom in allowed
            for symptom in symptoms
        ):
            rejected += 1
            continue

        if len(symptoms) != len(set(symptoms)):
            rejected += 1
            continue

        if not min_symptoms <= len(symptoms) <= max_symptoms:
            rejected += 1
            continue

        if len(symptoms) == 0:
            rejected += 1
            continue

        accepted.append({
            "prognosis": disease,

            "symptoms": json.dumps(
                sorted(symptoms),
                ensure_ascii=False
            ),

            "common_symptom_count": sum(
                symptom in common
                for symptom in symptoms
            ),

            "variable_symptom_count": sum(
                symptom in variable
                for symptom in symptoms
            ),

            "reference_source": reference["reference_source"],

            "reference_url": reference["reference_url"],

            "validation_status": "STRUCTURE_VALIDATION_ONLY"
        })

    return accepted, rejected


# ==========================================
# SAVE RECORDS SAFELY
# ==========================================

def save_records(accepted, output_file):

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    new_df = pd.DataFrame(accepted)

    if output_file.exists():

        old_df = pd.read_csv(output_file)

        columns = new_df.columns.tolist()

        for column in columns:
            if column not in old_df.columns:
                old_df[column] = pd.NA

        old_df = old_df[columns]

        combined = pd.concat(
            [old_df, new_df],
            ignore_index=True
        )

    else:
        combined = new_df.copy()

    combined["symptoms_key"] = combined[
        "symptoms"
    ].apply(
        lambda value: json.dumps(
            sorted(json.loads(value)),
            ensure_ascii=False
        )
    )

    before = len(combined)

    combined = combined.drop_duplicates(
        subset=["prognosis", "symptoms_key"]
    )

    duplicates_removed = before - len(combined)

    combined = combined.drop(
        columns=["symptoms_key"]
    )

    temporary_file = output_file.with_suffix(".tmp")

    combined.to_csv(
        temporary_file,
        index=False
    )

    temporary_file.replace(output_file)

    return len(combined), duplicates_removed


# ==========================================
# MAIN PROGRAM
# ==========================================

def main():

    print("=" * 50)
    print("GEMINI REFERENCE-GUIDED DATA GENERATOR")
    print("=" * 50)

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("ERROR: GEMINI_API_KEY is not set.")
        return

    try:

        reference = load_reference()

        output_file = get_output_file(reference)

        print("\nDisease:", reference["disease"])

        print(
            "Common symptoms:",
            len(reference["common_symptoms"])
        )

        print(
            "Variable symptoms:",
            len(reference["variable_symptoms"])
        )

        print(
            "Requested records:",
            reference["batch_size"]
        )

        print(
            "Allowed symptoms per record:",
            reference["min_symptoms"],
            "to",
            reference["max_symptoms"]
        )

        print("\nOutput:", output_file)

        client = genai.Client(
            api_key=api_key
        )

        print("\nRequesting records from Gemini...")

        records = generate_records(
            client,
            reference
        )

        print(
            "Records returned:",
            len(records)
        )

        accepted, rejected = validate_records(
            records,
            reference
        )

        print(
            "Structurally valid:",
            len(accepted)
        )

        print(
            "Rejected:",
            rejected
        )

        if not accepted:
            print("\nNo valid records to save.")
            return

        total, duplicates = save_records(
            accepted,
            output_file
        )

        print("\nGeneration completed.")

        print(
            "New valid records:",
            len(accepted)
        )

        print(
            "Duplicates removed:",
            duplicates
        )

        print(
            "Total unique saved records:",
            total
        )

        print(
            "Output file:",
            output_file
        )

        print("\nIMPORTANT:")
        print(
            "Structural validation is not medical validation."
        )

        print(
            "Expert review is required before model training."
        )

        print(
            "Original Training.csv and Testing.csv were not modified."
        )

    except Exception as error:

        print("\nERROR:", type(error).__name__)
        print(error)


if __name__ == "__main__":
    main()
