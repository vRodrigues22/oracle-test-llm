import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

print("OPENAI_API_KEY =", "OK" if os.getenv("OPENAI_API_KEY") else "MISSING")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

resp = client.responses.create(
    model="gpt-5.2",
    input="Responda apenas com OK."
)

print("Resposta:", resp.output_text)

