import os
import logging
import json
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv

# Indlæs miljøvariabler fra .env fil
load_dotenv()

app = FastAPI()

# Aktiver CORS hvis nødvendigt
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialisér OpenAI-klient
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Konfigurer logging
logging.basicConfig(level=logging.INFO)

def get_context_from_json(question: str, top_k: int = 3) -> str:
    try:
        with open("all_chunks.json", "r", encoding="utf-8") as f:
            all_chunks = json.load(f)
        chunks = [item["chunk"] for item in all_chunks if "chunk" in item][:top_k]
        return "\n\n".join(chunks)
    except Exception as e:
        logging.error(f"Context error: {e}")
        return ""

@app.post("/answer")
async def answer(request: Request):
    try:
        data = await request.json()
        question = data.get("question", "")
        logging.info(f"Received question: {question}")
        context = get_context_from_json(question)
        if not context:
            return {"answer": "Beklager, der opstod en fejl under hentning af konteksten."}

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Du er en hjælpsom AI-assistent for tandlæger. Brug kun viden fra konteksten nedenfor."},
                {"role": "user", "content": f"Kontekst: {context}\n\nSpørgsmål: {question}"},
            ],
        )
        return {"answer": response.choices[0].message.content}
    except Exception as e:
        logging.error(f"API error: {e}")
        return {"answer": "Beklager, der opstod en fejl under behandlingen af dit spørgsmål."}
