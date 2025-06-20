from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, validator
import json
import faiss
import numpy as np
from openai import OpenAI
import os
import sqlite3
import tiktoken
from datetime import datetime
import html
import logging
from contextlib import contextmanager
from typing import List

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configuration class
class Config:
    INDEX_FILE = os.getenv("FAISS_INDEX", "faiss.index")
    EMBEDDINGS_FILE = os.getenv("EMBEDDINGS_FILE", "embeddings.npy")
    CHUNKS_FILE = os.getenv("CHUNKS_FILE", "all_chunks.json")
    DB_PATH = os.getenv("KNOWLEDGE_DB", "knowledge.sqlite")
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", 3000))
    MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", 5))
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4")

# Global variables
chunks = []
index = None
tokenizer = tiktoken.get_encoding("cl100k_base")

# Load resources
@app.on_event("startup")
def initialize_resources():
    global chunks, index
    try:
        with open(Config.CHUNKS_FILE, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        index = faiss.read_index(Config.INDEX_FILE)
        logger.info(f"Resources loaded: {len(chunks)} chunks")
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        raise

@contextmanager
def get_db_connection():
    conn = None
    try:
        conn = sqlite3.connect(Config.DB_PATH)
        yield conn
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        if conn:
            conn.close()

def num_tokens(text: str) -> int:
    try:
        return len(tokenizer.encode(text))
    except Exception:
        return int(len(text.split()) * 1.3)

def get_top_chunks(question: str, k: int = Config.MAX_CHUNKS, max_tokens: int = Config.MAX_TOKENS) -> List[str]:
    try:
        response = client.embeddings.create(
            input=[question], model=Config.EMBEDDING_MODEL
        )
        vec = np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)
        scores, indices = index.search(vec, k)
        context_chunks = []
        total_tokens = 0
        for idx in indices[0]:
            if idx < 0 or idx >= len(chunks):
                continue
            chunk = chunks[idx].get("text", "")
            if not chunk:
                continue
            tok = num_tokens(chunk)
            if total_tokens + tok > max_tokens:
                break
            context_chunks.append(chunk)
            total_tokens += tok
        return context_chunks
    except Exception as e:
        logger.error(f"RAG error: {e}")
        return []

def get_rag_answer(question: str) -> str:
    if not question.strip():
        return "Please enter a valid question."
    context_chunks = get_top_chunks(question)
    if not context_chunks:
        return "⚠️ No relevant content found. Try rephrasing."
    prompt = (
        "You are Karin from AlignerService. You are a helpful assistant with experience in clear aligner support.\n"
        "Answer the question based only on the context below. If the context doesn't contain enough information\n"
        "or if more clinical expertise is needed, clearly state this.\n\n"
        f"Context:\n{chr(10).join(context_chunks)}\n\nQuestion: {question}\nAnswer:"
    )
    try:
        res = client.chat.completions.create(
            model=Config.CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        return "⚠️ Could not generate a response. Try again later."

def insert_into_db(ticket_id: str, question: str, answer: str) -> bool:
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tickets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket_id TEXT,
                    question TEXT,
                    answer TEXT,
                    source TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ticket_id, question)
                )
            ''')
            cursor.execute('''
                INSERT OR REPLACE INTO tickets (ticket_id, question, answer, source)
                VALUES (?, ?, ?, ?)
            ''', (ticket_id, question, answer, "RAG Assistant"))
            conn.commit()
            return True
    except Exception as e:
        logger.error(f"Insert DB error: {e}")
        return False

class AnswerRequest(BaseModel):
    ticketId: str
    question: str

    @validator('ticketId')
    def ticket_id_valid(cls, v):
        if not v.strip():
            raise ValueError("Ticket ID required")
        return v.strip()

    @validator('question')
    def question_valid(cls, v):
        if len(v.strip()) < 3:
            raise ValueError("Question must be at least 3 characters")
        return v.strip()

@app.post("/answer")
async def answer(req: AnswerRequest):
    answer_text = get_rag_answer(req.question)
    success = insert_into_db(req.ticketId, req.question, answer_text)
    return {"ticket_id": req.ticketId, "answer": answer_text, "success": success}

@app.get("/ui", response_class=HTMLResponse)
async def get_ui(ticketId: str = "", question: str = ""):
    return f"""
    <!DOCTYPE html>
    <html><head><title>AI Assistant</title></head><body>
    <h2>AlignerService AI Assistant</h2>
    <form method='post' action='/ui'>
    Ticket ID:<br><input name='ticketId' value='{html.escape(ticketId)}'><br><br>
    Your Question:<br><textarea name='question' rows='5'>{html.escape(question)}</textarea><br><br>
    <button type='submit'>Submit</button></form></body></html>"""

@app.post("/ui", response_class=HTMLResponse)
async def post_ui(request: Request):
    form = await request.form()
    ticket_id = str(form.get("ticketId", "")).strip()
    question = str(form.get("question", "")).strip()
    answer = get_rag_answer(question)
    insert_into_db(ticket_id, question, answer)
    return f"""
    <!DOCTYPE html>
    <html><head><title>Answer</title></head><body>
    <h2>AI Response</h2>
    <p><strong>Ticket ID:</strong> {html.escape(ticket_id)}</p>
    <p><strong>Question:</strong> {html.escape(question)}</p>
    <div style='background:#efefef;padding:10px;'>
    <strong>Answer:</strong><br>{html.escape(answer)}</div><br>
    <a href='/ui?ticketId={html.escape(ticket_id)}'>Ask another</a></body></html>"""

@app.get("/health")
async def health():
    return {"status": "ok", "chunks": len(chunks), "time": datetime.utcnow().isoformat()}
