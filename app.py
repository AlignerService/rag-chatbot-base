
import os
import sqlite3
import openai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from starlette.middleware.cors import CORSMiddleware

openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Payload(BaseModel):
    question: str
    ticketId: str = ""

@app.post("/answer")
async def answer(payload: Payload):
    try:
        conn = sqlite3.connect("knowledge.sqlite")
        cursor = conn.cursor()

        query = f"%{payload.question.lower()}%"
        cursor.execute("SELECT chunk FROM chunks WHERE chunk LIKE ?", (query,))
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            context = "Ingen relevant kontekst fundet i databasen."
        else:
            context = "\n\n".join(row[0] for row in rows[:5])

        messages = [
            {"role": "system", "content": "Du er en hjælpsom AI-assistent for en dansk aligner-supportservice. Giv præcise og venlige svar baseret på konteksten."},
            {"role": "user", "content": f"Spørgsmål: {payload.question}\n\nRelevant kontekst: {context}"}
        ]

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.3
        )
        return {"reply": response.choices[0].message.content.strip()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fejl: {str(e)}")
