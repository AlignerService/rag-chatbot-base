from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import openai
import os
import faiss
import numpy as np
import json
import html
import sqlite3
from datetime import datetime

# --- KONFIGURATION ---
EMBEDDINGS_FILE = "embeddings.npy"
INDEX_FILE = "faiss.index"
METADATA_FILE = "metadata.json"
DB_PATH = os.getenv("KNOWLEDGE_DB", "/Users/macpro/Dropbox/AlignerService/RAG:Database:aktiv/rag.sqlite3")

openai.api_key = os.getenv("OPENAI_API_KEY")

# --- LOAD FAISS + METADATA ---
index = faiss.read_index(INDEX_FILE)
embeddings = np.load(EMBEDDINGS_FILE)
with open(METADATA_FILE, "r") as f:
    metadata = json.load(f)

# --- FASTAPI SETUP ---
app = FastAPI()

# --- FUNKTIONER ---
def get_embedding(text):
    response = openai.Embedding.create(input=[text], model="text-embedding-ada-002")
    return np.array(response["data"][0]["embedding"], dtype=np.float32).reshape(1, -1)

def search(question, k=5):
    query_vector = get_embedding(question)
    D, I = index.search(query_vector, k)
    result_chunks = []
    for idx in I[0]:
        if idx < len(metadata):
            entry = metadata[idx]
            source = entry.get("source", "Ukendt kilde")
            title = entry.get("title", "Uden titel")
            text = entry.get("text", "")
            result_chunks.append({"title": title, "source": source, "text": text})
    return result_chunks

def generate_answer(question, context_chunks):
    context = "\n\n---\n\n".join([chunk['text'] for chunk in context_chunks])
    prompt = (
        "You are Karin from AlignerService. You are a helpful assistant with experience in clear aligner support.\n"
        "Answer the question based only on the context below. If the context doesn't contain enough information,\n"
        "say so and offer to forward the question to a clinical expert.\n\n"
        f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"⚠️ Error generating answer: {e}"

def insert_ticket(ticket_id, question, answer):
    try:
        conn = sqlite3.connect(DB_PATH)
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
        cursor.execute('''
            INSERT INTO tickets (ticket_id, question, answer, source)
            VALUES (?, ?, ?, ?)
        ''', (ticket_id, question, answer, "RAG"))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB insert error: {e}")

# --- DATAKLASSER ---
class AnswerRequest(BaseModel):
    ticketId: str
    question: str

# --- API ENDPOINTS ---
@app.post("/answer")
async def answer(req: AnswerRequest):
    chunks = search(req.question)
    answer = generate_answer(req.question, chunks)
    insert_ticket(req.ticketId, req.question, answer)
    return {
        "ticket_id": req.ticketId,
        "answer": answer,
        "source_chunks": chunks
    }

@app.get("/ui", response_class=HTMLResponse)
async def form_ui(ticketId: str = "", question: str = ""):
    return f"""
    <html>
    <head><title>AlignerService Assistant</title></head>
    <body>
        <h2>AlignerService AI Assistant</h2>
        <form action="/ui" method="post">
            <label>Ticket ID:</label><br>
            <input name="ticketId" value='{html.escape(ticketId)}'><br><br>
            <label>Question:</label><br>
            <textarea name="question" rows="4" cols="50">{html.escape(question)}</textarea><br><br>
            <button type="submit">Get Answer</button>
        </form>
    </body>
    </html>
    """

@app.post("/ui", response_class=HTMLResponse)
async def submit_form(request: Request):
    form = await request.form()
    ticket_id = form.get("ticketId", "")
    question = form.get("question", "")
    chunks = search(question)
    answer = generate_answer(question, chunks)
    insert_ticket(ticket_id, question, answer)
    return f"""
    <html>
    <head><title>AlignerService Assistant</title></head>
    <body>
        <h2>Answer</h2>
        <p><strong>Ticket ID:</strong> {html.escape(ticket_id)}</p>
        <p><strong>Question:</strong> {html.escape(question)}</p>
        <p><strong>Answer:</strong><br>{html.escape(answer)}</p>
        <br><a href="/ui?ticketId={html.escape(ticket_id)}">Ask another</a>
    </body>
    </html>
    """

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "chunks": len(metadata),
        "timestamp": datetime.now().isoformat()
    }
