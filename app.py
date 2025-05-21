from fastapi import FastAPI, Request
from pydantic import BaseModel
import sqlite3
import dropbox
import os
import requests
from datetime import datetime

app = FastAPI()

# Dropbox upload opsætning
DROPBOX_TOKEN = os.getenv("DROPBOX_TOKEN")
DROPBOX_DB_PATH = "/knowledge.sqlite"
TEMP_LOCAL_DB_PATH = "/tmp/knowledge.sqlite"
dbx = dropbox.Dropbox(DROPBOX_TOKEN)

# ZoHo OAuth credentials (hentes nu fra Render miljøvariabler)
CLIENT_ID = os.getenv("ZOHO_CLIENT_ID")
CLIENT_SECRET = os.getenv("ZOHO_CLIENT_SECRET")
REFRESH_TOKEN = os.getenv("ZOHO_REFRESH_TOKEN")
TOKEN_URL = "https://accounts.zoho.eu/oauth/v2/token"
API_URL = "https://desk.zoho.eu/api/v1"

class UpdateRequest(BaseModel):
    ticketId: str

@app.get("/")
def read_root():
    return {"message": "RAG API is live."}

def get_valid_access_token():
    payload = {
        "refresh_token": REFRESH_TOKEN,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_type": "refresh_token"
    }
    response = requests.post(TOKEN_URL, data=payload)
    token_data = response.json()
    return token_data.get("access_token")

def get_ticket_thread(ticket_id):
    token = get_valid_access_token()
    headers = {"Authorization": f"Zoho-oauthtoken {token}"}
    url = f"{API_URL}/tickets/{ticket_id}/conversations"
    response = requests.get(url, headers=headers)
    return response.json()

def store_ticket_thread(ticket_id, thread_data):
    conn = sqlite3.connect(TEMP_LOCAL_DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ticket_threads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_id TEXT,
            sender TEXT,
            content TEXT,
            time TEXT
        )
    ''')

    for item in thread_data.get("data", []):
        sender = item.get("fromEmail") or item.get("sender") or "unknown"
        content = item.get("content") or ""
        timestamp = item.get("createdTime") or datetime.utcnow().isoformat()
        cursor.execute(
            "INSERT INTO ticket_threads (ticket_id, sender, content, time) VALUES (?, ?, ?, ?)",
            (ticket_id, sender, content, timestamp)
        )

    conn.commit()
    conn.close()

    with open(TEMP_LOCAL_DB_PATH, "rb") as f:
        dbx.files_upload(f.read(), DROPBOX_DB_PATH, mode=dropbox.files.WriteMode.overwrite)

@app.post("/update_ticket")
async def update_ticket(req: Request):
    data = await req.json()
    ticket_id = data.get("ticketId")
    if not ticket_id:
        return {"error": "Missing ticketId"}

    thread = get_ticket_thread(ticket_id)
    store_ticket_thread(ticket_id, thread)
    return {"status": "Ticket thread saved", "ticketId": ticket_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
