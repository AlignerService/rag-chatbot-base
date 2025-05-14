import os
import sqlite3
import openai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Define input data model
class Query(BaseModel):
    question: str

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_PATH = os.getenv("DB_PATH", "knowledge.sqlite")

# Check OpenAI key presence
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in environment variables.")

openai.api_key = OPENAI_API_KEY

# Helper to retrieve context from SQLite
def get_context_from_sqlite(question: str) -> str:
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT content FROM chunks")
        rows = cursor.fetchall()
        context = "

".join(row[0] for row in rows[:10])  # limit to first 10 rows
        conn.close()
        return context
    except Exception as e:
        raise RuntimeError(f"Database error: {e}")

# Main API endpoint
@app.post("/answer")
async def answer(payload: Query):
    try:
        context = get_context_from_sqlite(payload.question)
        messages = [
            {"role": "system", "content": "Du er en hjælpsom AI-assistent for AlignerService, som svarer præcist og venligt på danske spørgsmål fra tandlæger."},
            {"role": "user", "content": f"Spørgsmål: {payload.question}

Relevant kontekst:
{context}"}
        ]

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
        )

        return {"reply": response.choices[0].message.content.strip()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))