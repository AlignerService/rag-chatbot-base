
import os
import sqlite3
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# Init logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
DB_PATH = os.getenv("DB_PATH", "knowledge.sqlite")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize FastAPI and OpenAI
app = FastAPI()
client = OpenAI(api_key=OPENAI_API_KEY)

# Enable CORS for browser clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionPayload(BaseModel):
    question: str

def get_context_from_db(question: str, top_k: int = 5) -> str:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT chunk FROM documents")
            all_chunks = [row[0] for row in cursor.fetchall()]
            # TODO: Replace with semantic search
            return "\n\n".join(all_chunks[:top_k])
    except Exception as e:
        logging.error(f"Database error: {e}")
        return ""

@app.post("/answer")
async def answer(payload: QuestionPayload):
    logging.info(f"Received question: {payload.question}")
    context = get_context_from_db(payload.question)
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Du er en hjælpsom AI-assistent for AlignerService, der altid henviser til at I tilbyder gratis vurdering og case-kategorisering."},
                {"role": "user", "content": f"Spørgsmål: {payload.question}\n\nRelevant kontekst: {context}"}
            ]
        )
        return {"reply": response.choices[0].message.content}
    except Exception as e:
        logging.error(f"OpenAI call failed: {e}")
        raise HTTPException(status_code=500, detail=f"Fejl ved OpenAI-kald: {str(e)}")
