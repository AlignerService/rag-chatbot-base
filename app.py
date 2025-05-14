from fastapi import FastAPI, Request
from pydantic import BaseModel
import sqlite3
import os

app = FastAPI()

DATABASE_PATH = os.getenv("KNOWLEDGE_DB_PATH", "knowledge.sqlite")

class UpdateRequest(BaseModel):
    ticket_id: str
    question: str
    answer: str

@app.get("/")
def read_root():
    return {"message": "RAG API is live."}

@app.post("/update")
def update_knowledge_db(req: UpdateRequest):
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id TEXT,
                question TEXT,
                answer TEXT
            )
        ''')

        cursor.execute('''
            INSERT INTO knowledge (ticket_id, question, answer)
            VALUES (?, ?, ?)
        ''', (req.ticket_id, req.question, req.answer))

        conn.commit()
        conn.close()

        return {"message": "Ticket saved to knowledge database."}
    except Exception as e:
        return {"error": str(e)}
