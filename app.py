
import os
import sqlite3
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
import openai

app = FastAPI()

DB_PATH = "knowledge.sqlite"
openai.api_key = os.getenv("OPENAI_API_KEY")

class ChatRequest(BaseModel):
    question: str

def get_chunks_from_db() -> List[str]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT content FROM chunks")
    chunks = [row[0] for row in cursor.fetchall()]
    conn.close()
    return chunks

def build_prompt(question: str, context_chunks: List[str]) -> str:
    context = "\n---\n".join(context_chunks[:10])  # begræns til 10 chunks
    prompt = f"""Besvar spørgsmålet baseret på nedenstående viden. Hvis du ikke ved det, så sig det ærligt.

Viden:
{context}

Spørgsmål:
{question}

Svar:
"""
    return prompt

@app.post("/answer")
async def answer(req: ChatRequest):
    question = req.question
    chunks = get_chunks_from_db()
    prompt = build_prompt(question, chunks)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Du er en hjælpsom AI-assistent for AlignerService. Vær præcis og brug vores tone of voice."},
                {"role": "user", "content": prompt}
            ]
        )
        reply = response.choices[0].message.content.strip()
    except Exception as e:
        reply = f"Fejl ved OpenAI-kald: {str(e)}"

    return {"reply": reply}
