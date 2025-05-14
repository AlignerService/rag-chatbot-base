
import os
import sqlite3
import openai
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import requests

app = FastAPI()
logging.basicConfig(level=logging.INFO)

# Miljøvariabler
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ZOHO_CLIENT_ID = os.getenv("ZOHO_CLIENT_ID")
ZOHO_CLIENT_SECRET = os.getenv("ZOHO_CLIENT_SECRET")
ZOHO_REFRESH_TOKEN = os.getenv("ZOHO_REFRESH_TOKEN")
DATABASE_PATH = "knowledge.sqlite"

openai.api_key = OPENAI_API_KEY

class QueryPayload(BaseModel):
    question: str
    ticketId: str

def get_context_from_sqlite(question: str, k: int = 5) -> List[str]:
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT content FROM chunks ORDER BY RANDOM() LIMIT ?", (k,))
    rows = cursor.fetchall()
    conn.close()
    return [row[0] for row in rows]

def refresh_zoho_token() -> str:
    url = "https://accounts.zoho.eu/oauth/v2/token"
    params = {
        "refresh_token": ZOHO_REFRESH_TOKEN,
        "client_id": ZOHO_CLIENT_ID,
        "client_secret": ZOHO_CLIENT_SECRET,
        "grant_type": "refresh_token",
    }
    response = requests.post(url, params=params)
    if response.status_code == 200:
        access_token = response.json().get("access_token")
        return access_token
    else:
        logging.error("Failed to refresh ZoHo token")
        return None

@app.post("/answer")
def answer(payload: QueryPayload):
    try:
        context_chunks = get_context_from_sqlite(payload.question)
        context_str = "\n---\n".join(context_chunks)

        messages = [
            {"role": "system", "content": "Du er en hjælpsom AI-assistent for AlignerService. Brug kun information fra konteksten."},
            {"role": "user", "content": f"Spørgsmål: {payload.question}

Relevant kontekst:
{context_str}"},
        ]

        completion = openai.chat.completions.create(
            model="gpt-4",
            messages=messages,
        )
        reply = completion.choices[0].message.content
        return {"reply": reply}

    except Exception as e:
        logging.exception("Fejl ved OpenAI-kald")
        raise HTTPException(status_code=500, detail=f"Fejl ved OpenAI-kald: {str(e)}")
