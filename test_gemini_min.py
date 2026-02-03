import os
from dotenv import load_dotenv
from google import genai

# Carrega .env da pasta atual (raiz do projeto)
load_dotenv()

print("GEMINI_API_KEY =", "OK" if os.getenv("GEMINI_API_KEY") else "MISSING")

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

resp = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="Responda apenas com OK."
)

print("Resposta:", resp.text)
