import os
import json
import faiss
import numpy as np
import sqlite3
import tiktoken
import openai
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from datetime import datetime

# OpenAI API-nøgle
openai.api_key = os.getenv("OPENAI_API_KEY")

# Konfiguration
EMBEDDING_MODEL = "text-embedding-ada-002"
INDEX_FILE = "faiss.index"
EMBEDDINGS_FILE = "embeddings.npy"
METADATA_FILE = "metadata.json"
DATABASE_PATH = os.getenv("KNOWLEDGE_DB", "/Users/macpro/Dropbox/AlignerService/RAG:Database:aktiv/rag.sqlite3")

# Tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

# Indlæs metadata
with open(METADATA_FILE, "r") as f:
    metadata = json.load(f)

# Indlæs FAISS-indeks
index = faiss.read_index(INDEX_FILE)

app = FastAPI()

def get_embedding(text):
    response = openai.Embedding.create(input=[text], model=EMBEDDING_MODEL)
    return np.array(response["data"][0]["embedding"], dtype=np.float32)

def search(question, k=5, max_tokens=3000):
    query_vector = get_embedding(question).reshape(1, -1)
    scores, indices = index.search(query_vector, k)
    
    context_chunks = []
    total_tokens = 0

    for idx in indices[0]:
        if idx < 0 or idx >= len(metadata):
            continue

        entry = metadata[idx]
        chunk_text = entry.get("text", "")
        if not chunk_text:
            continue

        chunk_tokens = len(tokenizer.encode(chunk_text))
        if total_tokens + chunk_tokens > max_tokens:
            break

        context_chunks.append(chunk_text)
        total_tokens += chunk_tokens

    return context_chunks

def generate_answer(question, context_chunks):
    context = "\n\n---\n\n".join(context_chunks)
    prompt = (
        "You are Karin from AlignerService. You are a helpful assistant with experience in clear aligner support.\n"
        "Answer the question based only on the context below. If the context doesn't contain enough information "
        "or if more clinical expertise is needed, clearly state this.\n\n"
        f"Context:\n{context}\n\n---\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=500
    )

    return response["choices"][0]["message"]["content"].strip()

def save_to_database(ticket_id, question, answer):
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_id TEXT,
            question TEXT,
            answer TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticket_id, question)
        )
    ''')

    cursor.execute('''
        INSERT OR REPLACE INTO tickets (ticket_id, question, answer)
        VALUES (?, ?, ?)
    ''', (ticket_id, question, answer))

    conn.commit()
    conn.close()

@app.post("/answer")
async def answer(request: Request):
    data = await request.json()
    ticket_id = data.get("ticketId", "").strip()
    question = data.get("question", "").strip()

    if not ticket_id or not question:
        return {"error": "Missing ticketId or question"}

    chunks = search(question)
    if not chunks:
        answer_text = "⚠️ No relevant context found. Please try rephrasing your question."
    else:
        answer_text = generate_answer(question, chunks)

    save_to_database(ticket_id, question, answer_text)

    return {
        "ticketId": ticket_id,
        "question": question,
        "answer": answer_text
    }

@app.get("/ui", response_class=HTMLResponse)
async def get_ui():
    return """
    <html>
        <head>
            <title>AlignerService Assistant</title>
        </head>
        <body>
            <h1>Ask Karin – your aligner assistant</h1>
            <form action="/ui" method="post">
                <label>Ticket ID:</label><br>
                <input type="text" name="ticketId"><br><br>
                <label>Question:</label><br>
                <textarea name="question" rows="4" cols="50"></textarea><br><br>
                <input type="submit" value="Submit">
            </form>
        </body>
    </html>
    """

@app.post("/ui", response_class=HTMLResponse)
async def post_ui(request: Request):
    form = await request.form()
    ticket_id = form.get("ticketId", "").strip()
    question = form.get("question", "").strip()

    if not ticket_id or not question:
        return HTMLResponse("<p>Missing ticket ID or question</p>")

    chunks = search(question)
    if not chunks:
        answer = "⚠️ No relevant context found. Please try rephrasing your question."
    else:
        answer = generate_answer(question, chunks)

    save_to_database(ticket_id, question, answer)

    return f"""
    <html>
        <head>
            <title>Answer</title>
        </head>
        <body>
            <h2>Answer:</h2>
            <p><strong>Ticket ID:</strong> {ticket_id}</p>
            <p><strong>Question:</strong> {question}</p>
            <p><strong>Answer:</strong> {answer}</p>
            <a href="/ui">Ask another question</a>
        </body>
    </html>
    """
