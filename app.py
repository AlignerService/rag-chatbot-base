from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import json
import faiss
import numpy as np
import openai
import os
import sqlite3
import tiktoken
from datetime import datetime

app = FastAPI()

# API-nøgle
openai.api_key = os.getenv("OPENAI_API_KEY")

# Filstier
INDEX_FILE = "faiss.index"
EMBEDDINGS_FILE = "embeddings.npy"
CHUNKS_FILE = "all_chunks.json"
DB_PATH = os.getenv("KNOWLEDGE_DB", "/Users/macpro/Dropbox/AlignerService/RAG:Database:aktiv/knowledge.sqlite")

# Tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

# Hent chunks
with open(CHUNKS_FILE, "r") as f:
    chunks = json.load(f)

# Hent FAISS
index = faiss.read_index(INDEX_FILE)
embeddings = np.load(EMBEDDINGS_FILE)


def num_tokens(text):
    return len(tokenizer.encode(text))


def get_top_chunks(question, k=5, max_tokens=3000):
    try:
        response = openai.Embedding.create(input=[question], model="text-embedding-3-small")
        vec = np.array(response["data"][0]["embedding"], dtype=np.float32).reshape(1, -1)
        scores, indices = index.search(vec, k)

        context_chunks = []
        total_tokens = 0
        for idx in indices[0]:
            if 0 <= idx < len(chunks):
                chunk = chunks[idx]["text"]
                chunk_tokens = num_tokens(chunk)
                if total_tokens + chunk_tokens < max_tokens:
                    context_chunks.append(chunk)
                    total_tokens += chunk_tokens
        return context_chunks
    except Exception as e:
        print(f"⚠️ Fejl i get_top_chunks: {e}")
        return []


def get_rag_answer(question):
    context_chunks = get_top_chunks(question)
    if not context_chunks:
        return "⚠️ No relevant context found."

    context = "\n\n---\n\n".join(context_chunks)
    prompt = (
        "You are Karin from AlignerService. You are a helpful assistant with experience in clear aligner support.\n"
        "Answer the question based only on the context below. If more clinical expertise is needed, say so.\n\n"
        f"{context}\n\n---\n\nQuestion: {question}\nAnswer:"
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"⚠️ No answer received. Error: {e}"


def insert_into_db(ticket_id, question, answer):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id TEXT UNIQUE,
                question TEXT,
                answer TEXT,
                source TEXT
            )
        ''')
        cursor.execute('''
            INSERT OR IGNORE INTO tickets (ticket_id, question, answer, source)
            VALUES (?, ?, ?, ?)
        ''', (ticket_id, question, answer, "RAG Assistant"))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"⚠️ Fejl ved DB-insert: {e}")


class AnswerRequest(BaseModel):
    ticketId: str
    question: str


@app.post("/answer")
async def answer(req: AnswerRequest):
    answer_text = get_rag_answer(req.question)
    insert_into_db(req.ticketId, req.question, answer_text)
    return {"answer": answer_text}


@app.get("/ui", response_class=HTMLResponse)
async def get_ui(ticketId: str = "", question: str = ""):
    return f"""
    <html>
    <head><title>AI Assistant</title></head>
    <body>
        <h2>AlignerService Assistant</h2>
        <form method="post" action="/ui">
            <input type="text" name="ticketId" placeholder="Ticket ID" value="{ticketId}" /><br />
            <textarea name="question" rows="4" cols="50" placeholder="Your question here">{question}</textarea><br />
            <button type="submit">Get Answer</button>
        </form>
    </body>
    </html>
    """


@app.post("/ui", response_class=HTMLResponse)
async def post_ui(request: Request):
    form = await request.form()
    ticket_id = form.get("ticketId", "")
    question = form.get("question", "")
    answer = get_rag_answer(question)
    insert_into_db(ticket_id, question, answer)

    return f"""
    <html>
    <body>
        <h2>AlignerService Assistant</h2>
        <p><strong>Ticket ID:</strong> {ticket_id}</p>
        <p><strong>Question:</strong> {question}</p>
        <p><strong>Answer:</strong> {answer}</p>
        <a href="/ui?ticketId={ticket_id}">Ask another question</a>
    </body>
    </html>
    """


@app.post("/update_ticket")
async def update_ticket(req: AnswerRequest):
    answer = get_rag_answer(req.question)
    insert_into_db(req.ticketId, req.question, answer)
    return {"ticket_id": req.ticketId, "answer": answer}
