
from fastapi import FastAPI, Request
from pydantic import BaseModel
import sqlite3
from datetime import datetime

app = FastAPI()

def init_db():
    conn = sqlite3.connect("tickets.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            ticket_id TEXT PRIMARY KEY,
            subject TEXT,
            message TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

class Ticket(BaseModel):
    ticket_id: str
    subject: str
    message: str
    created_at: str = datetime.utcnow().isoformat()

@app.post("/save_ticket")
async def save_ticket(ticket: Ticket):
    try:
        conn = sqlite3.connect("tickets.db")
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO tickets (ticket_id, subject, message, created_at)
            VALUES (?, ?, ?, ?)
        """, (ticket.ticket_id, ticket.subject, ticket.message, ticket.created_at))
        conn.commit()
        conn.close()
        return {"status": "success", "ticket_id": ticket.ticket_id}
    except Exception as e:
        return {"status": "error", "message": str(e)}

class Question(BaseModel):
    question: str

@app.post("/answer")
async def answer(question: Question):
    return {
        "answer": f"Dette er et placeholder svar på: '{question.question}'",
        "source": "demo-mode"
    }

@app.get("/")
def read_root():
    return {"status": "ok", "message": "AlignerService RAG is running"}

@app.get("/healthz")
def health_check():
    return {"ok": True}
