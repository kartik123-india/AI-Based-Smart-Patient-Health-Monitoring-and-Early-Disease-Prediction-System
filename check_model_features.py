
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent

for filename in ["disease_model.joblib", "disease_model_augmented.joblib"]:
    path = BASE_DIR / "models" / filename
    obj = joblib.load(path)

    if isinstance(obj, dict):
        model = obj["model"]
        features = obj["feature_columns"]
    else:
        model = obj
        features = list(model.feature_names_in_)

    print("\n" + "=" * 45)
    print(filename)
    print("Feature count:", len(features))
    print("high_fever index:", features.index("high_fever")
          if "high_fever" in features else "MISSING")
    print("runny_nose index:", features.index("runny_nose")
          if "runny_nose" in features else "MISSING")
    print("Classes:", len(model.classes_))

    if hasattr(model, "n_features_in_"):
        print("Model expects features:", model.n_features_in_)

print("\nFeature check complete.")
