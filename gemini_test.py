
import os
from google import genai

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise SystemExit(
        "GEMINI_API_KEY is missing. Set it in CMD first."
    )

client = genai.Client(api_key=api_key)

try:
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents="Reply with exactly: Gemini connection successful."
    )

    print("Gemini response:")
    print(response.text)

except Exception as error:
    print("Gemini request failed.")
    print(type(error).__name__)
    print(error)
