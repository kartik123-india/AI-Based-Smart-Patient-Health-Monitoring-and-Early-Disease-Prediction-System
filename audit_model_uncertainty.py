from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

BASE_DIR = Path(__file__).resolve().parent
TEST_PATH = BASE_DIR / "dataset" / "Testing.csv"
MODEL_PATH = BASE_DIR / "models" / "disease_model_augmented.joblib"

warnings.filterwarnings("ignore")

def main():
    if not TEST_PATH.exists() or not MODEL_PATH.exists():
        print("Testing.csv or disease_model_augmented.joblib was not found.")
        return

    test_df = pd.read_csv(TEST_PATH)
    model = joblib.load(MODEL_PATH)

    if isinstance(model, dict):
        model = model.get("model")

    if model is None or not hasattr(model, "predict_proba"):
        print("Loaded model does not support predict_proba.")
        return

    features = list(model.feature_names_in_)
    missing = [col for col in features if col not in test_df.columns]
    if missing:
        print("Missing test features:", missing)
        return

    X = test_df[features]
    probabilities = model.predict_proba(X)
    classes = np.asarray(model.classes_).astype(str)

    ranked = np.argsort(probabilities, axis=1)[:, ::-1]
    top1_idx = ranked[:, 0]
    top2_idx = ranked[:, 1]
    top1_score = probabilities[np.arange(len(test_df)), top1_idx]
    top2_score = probabilities[np.arange(len(test_df)), top2_idx]
    margin = top1_score - top2_score
    predicted = classes[top1_idx]

    report = pd.DataFrame({
        "Actual": test_df["prognosis"].astype(str).str.strip()
                  if "prognosis" in test_df.columns else "Unknown",
        "Predicted": pd.Series(predicted).str.strip(),
        "Top1_Score_Percent": np.round(top1_score * 100, 2),
        "Top2_Score_Percent": np.round(top2_score * 100, 2),
        "Top1_Top2_Margin_Percent": np.round(margin * 100, 2),
        "Correct": (
            pd.Series(predicted).str.strip().values
            == test_df["prognosis"].astype(str).str.strip().values
        ) if "prognosis" in test_df.columns else None
    })

    print("=" * 72)
    print("MODEL SCORE AND UNCERTAINTY AUDIT")
    print("=" * 72)
    print("Test records:", len(test_df))
    if "prognosis" in test_df.columns:
        print("Accuracy:", round(accuracy_score(report["Actual"], report["Predicted"]), 4))

    print("\nTOP-1 SCORE SUMMARY (%):")
    print(report["Top1_Score_Percent"].describe().round(2).to_string())

    print("\nTOP-1 VS TOP-2 MARGIN SUMMARY (%):")
    print(report["Top1_Top2_Margin_Percent"].describe().round(2).to_string())

    if "Correct" in report.columns:
        print("\nSCORE SUMMARY FOR CORRECT VS INCORRECT PREDICTIONS:")
        print(
            report.groupby("Correct")[
                ["Top1_Score_Percent", "Top2_Score_Percent",
                 "Top1_Top2_Margin_Percent"]
            ].agg(["count", "mean", "min", "max"]).round(2).to_string()
        )

    print("\nALL TEST RECORDS:")
    print(report.to_string(index=False))

    output = BASE_DIR / "model_uncertainty_audit.csv"
    report.to_csv(output, index=False)
    print("\nSaved detailed results to:", output)
    print("\nIMPORTANT: Random Forest scores are not calibrated medical probabilities.")
    print("This audit does not establish clinical accuracy or a safe diagnosis threshold.")

if __name__ == "__main__":
    main()
