
from fastapi import FastAPI, Request
from pydantic import BaseModel
import sqlite3
from datetime import datetime

app = FastAPI()

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
