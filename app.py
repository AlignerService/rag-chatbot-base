import os
import json
import sqlite3
import logging
import numpy as np
import asyncio
import tiktoken
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import openai
import faiss

# Opsæt logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

app = FastAPI()

# Miljøvariabler til konfiguration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_PATH = os.getenv("KNOWLEDGE_DB", "/Users/macpro/Dropbox/AlignerService/RAG:Database:aktiv/rag.sqlite3")
INDEX_FILE = os.getenv("FAISS_INDEX_FILE", "faiss.index")
METADATA_FILE = os.getenv("METADATA_FILE", "metadata.json")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4")
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0.2))

openai.api_key = OPENAI_API_KEY

# Init database med tabel hvis ikke eksisterer
def init_db():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tickets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket_id TEXT,
                    question TEXT,
                    answer TEXT,
                    source TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
            logging.info("Database initialized successfully.")
    except Exception as e:
        logging.error(f"DB initialization error: {e}")

init_db()

# Load FAISS index og metadata med try-except
try:
    index = faiss.read_index(INDEX_FILE)
    logging.info(f"FAISS index loaded from {INDEX_FILE}")
except Exception as e:
    logging.error(f"Failed to load FAISS index: {e}")
    index = None

try:
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    logging.info(f"Metadata loaded from {METADATA_FILE}")
except Exception as e:
    logging.error(f"Failed to load metadata: {e}")
    metadata = []

class AnswerRequest(BaseModel):
    ticketId: str = Field(..., max_length=100)
    question: str = Field(..., max_length=1000)

def trim_context(context_chunks, max_tokens=6000):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens_used = 0
    trimmed_chunks = []
    for chunk in context_chunks:
        chunk_tokens = len(tokenizer.encode(chunk['text']))
        if tokens_used + chunk_tokens > max_tokens:
            break
        trimmed_chunks.append(chunk)
        tokens_used += chunk_tokens
    return trimmed_chunks

def get_top_chunks(question: str, top_k: int = 5):
    if index is None or not metadata:
        logging.warning("Index or metadata not loaded, cannot search.")
        return []

    try:
        response = openai.Embedding.create(input=[question], model="text-embedding-ada-002")
        query_vector = response["data"][0]["embedding"]
    except Exception as e:
        logging.error(f"OpenAI embedding error: {e}")
        return []

    D, I = index.search(np.array([query_vector]).astype('float32'), top_k)
    result_chunks = []
    for idx in I[0]:
        if idx < len(metadata):
            entry = metadata[idx]
            result_chunks.append(entry)
    return result_chunks

def generate_answer(question: str, context_chunks: list):
    if not context_chunks:
        return "Der findes ikke nok information til at besvare spørgsmålet. Skal jeg sende spørgsmålet videre til en klinisk ekspert?"

    trimmed_chunks = trim_context(context_chunks, max_tokens=6000)
    context = "\n\n---\n\n".join([chunk['text'] for chunk in trimmed_chunks])

    prompt = (
        "Du er Karin fra AlignerService, en erfaren klinisk rådgiver.\n"
        "Svar så informativt som muligt baseret på følgende kontekst.\n"
        "Hvis spørgsmålet kræver klinisk ekspertise ud over din kapacitet, så sig det tydeligt og tilbyd at involvere en klinisk rådgiver.\n\n"
        f"Kontekst:\n{context}\n\nSpørgsmål:\n{question}\n\nSvar:"
    )

    try:
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=300,
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logging.error(f"OpenAI chat completion error: {e}")
        return f"Der opstod en fejl ved generering af svar: {e}"

def save_to_db(ticket_id, question, answer_text):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO tickets (ticket_id, question, answer, source)
                VALUES (?, ?, ?, ?)
            ''', (ticket_id, question, answer_text, "RAG"))
            conn.commit()
            logging.info(f"Saved ticket {ticket_id} to database")
    except Exception as e:
        logging.error(f"DB insert error: {e}")
        raise

@app.post("/api/answer")
async def answer(request: AnswerRequest):
    logging.info(f"Received question for ticketId {request.ticketId}")

    chunks = await asyncio.to_thread(get_top_chunks, request.question, top_k=5)
    answer_text = await asyncio.to_thread(generate_answer, request.question, chunks)

    try:
        await asyncio.to_thread(save_to_db, request.ticketId, request.question, answer_text)
    except Exception as e:
        logging.error(f"DB insert error: {e}")
        raise HTTPException(status_code=500, detail="Database error")

    return {"answer": answer_text}
