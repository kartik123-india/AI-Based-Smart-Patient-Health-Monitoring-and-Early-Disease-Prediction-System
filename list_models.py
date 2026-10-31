import os
from google import genai

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise SystemExit("GEMINI_API_KEY is not set in this CMD window.")

try:
    client = genai.Client(api_key=api_key)

    print("Models supporting text generation:\n")

    for model in client.models.list():
        if "generateContent" in (model.supported_actions or []):
            print(model.name)

except Exception as error:
    print("Could not list models.")
    print(type(error).__name__)
    print(error)