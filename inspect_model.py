
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent

for filename in [
    "disease_model.joblib",
    "disease_model_augmented.joblib",
]:
    path = BASE_DIR / "models" / filename

    print(f"\n--- {filename} ---")

    if not path.exists():
        print("File not found")
        continue

    model = joblib.load(path)

    print("Object type:", type(model).__name__)

    if isinstance(model, dict):
        print("Dictionary keys:", list(model.keys()))

        for key, value in model.items():
            print(f"  {key}: {type(value).__name__}")
    else:
        print("Has feature_names_in_:", hasattr(model, "feature_names_in_"))
        print("Has predict:", hasattr(model, "predict"))
