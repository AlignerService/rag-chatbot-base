
import os
import sqlite3
import openai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

DB_PATH = "knowledge.sqlite"
openai.api_key = os.environ.get("OPENAI_API_KEY")

class Payload(BaseModel):
    question: str

def get_context_from_db(question: str, top_k: int = 5) -> str:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT chunk FROM documents")
    all_chunks = [row[0] for row in cursor.fetchall()]
    conn.close()

    # Simpel søgning: returner de første 'top_k' chunks som kontekst
    return "\n\n".join(all_chunks[:top_k])

@app.post("/answer")
async def answer(payload: Payload):
    try:
        context = get_context_from_db(payload.question)
        messages = [
            {"role": "system", "content": "Du er en hjælpsom AI-assistent, der svarer tandlæger professionelt og præcist baseret på eksisterende viden fra tidligere sager."},
            {"role": "user", "content": f"Spørgsmål: {payload.question}\n\nRelevant kontekst:\n{context}"}
        ]

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=messages
        )

        return {"reply": response.choices[0].message.content.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
