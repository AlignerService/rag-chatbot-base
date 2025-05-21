from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import sqlite3
import os
import requests
from datetime import datetime
import logging
import html
import openai

app = FastAPI()

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Miljøvariabler
DROPBOX_ACCESS_TOKEN = os.getenv("DROPBOX_ACCESS_TOKEN")
DROPBOX_REFRESH_TOKEN = os.getenv("DROPBOX_REFRESH_TOKEN")
DROPBOX_CLIENT_ID = os.getenv("DROPBOX_CLIENT_ID")
DROPBOX_CLIENT_SECRET = os.getenv("DROPBOX_CLIENT_SECRET")
DROPBOX_DB_PATH = os.getenv("DROPBOX_DB_PATH", "/knowledge.sqlite")
TEMP_LOCAL_DB_PATH = os.getenv("LOCAL_DB_PATH", "/tmp/knowledge.sqlite")

CLIENT_ID = os.getenv("ZOHO_CLIENT_ID")
CLIENT_SECRET = os.getenv("ZOHO_CLIENT_SECRET")
REFRESH_TOKEN = os.getenv("ZOHO_REFRESH_TOKEN")
TOKEN_URL = "https://accounts.zoho.eu/oauth/v2/token"
API_URL = "https://desk.zoho.eu/api/v1"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Tjek for manglende miljøvariabler
required_env_vars = {
    "DROPBOX_REFRESH_TOKEN": DROPBOX_REFRESH_TOKEN,
    "DROPBOX_CLIENT_ID": DROPBOX_CLIENT_ID,
    "DROPBOX_CLIENT_SECRET": DROPBOX_CLIENT_SECRET,
    "ZOHO_CLIENT_ID": CLIENT_ID,
    "ZOHO_CLIENT_SECRET": CLIENT_SECRET,
    "ZOHO_REFRESH_TOKEN": REFRESH_TOKEN,
    "OPENAI_API_KEY": OPENAI_API_KEY
}
missing = [key for key, value in required_env_vars.items() if not value]
if missing:
    raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

class UpdateRequest(BaseModel):
    ticketId: str

class AnswerRequest(BaseModel):
    ticketId: str
    question: str

@app.get("/")
def read_root():
    return {"message": "✅ FastAPI is working."}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/tickets")
def get_ticket(ticket_id: str):
    if not ticket_id or not ticket_id.strip().isalnum():
        raise HTTPException(status_code=400, detail="Invalid ticketId format")

    try:
        conn = sqlite3.connect(TEMP_LOCAL_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT sender, content, time FROM ticket_threads WHERE ticket_id = ? ORDER BY time ASC", (ticket_id,))
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            raise HTTPException(status_code=404, detail="No data found for this ticket")

        return [{"sender": row[0], "content": row[1], "time": row[2]} for row in rows]

    except Exception as e:
        logger.error(f"❌ Error reading from SQLite: {e}")
        raise HTTPException(status_code=500, detail="Failed to read ticket data")

def get_valid_access_token():
    payload = {
        "refresh_token": REFRESH_TOKEN,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_type": "refresh_token"
    }
    try:
        response = requests.post(TOKEN_URL, data=payload, timeout=10)
        logger.info(f"🔍 RAW RESPONSE TEXT: {response.text}")
        response.raise_for_status()
        token_data = response.json()
        logger.info(f"🔍 TOKEN RESPONSE: {token_data}")
        access_token = token_data.get("access_token")
        if not access_token:
            raise Exception("No access token in response")
        return access_token
    except Exception as e:
        logger.error(f"❌ Error obtaining ZoHo token: {e}")
        raise HTTPException(status_code=500, detail="Failed to get ZoHo access token")

def get_dropbox_access_token():
    payload = {
        "grant_type": "refresh_token",
        "refresh_token": DROPBOX_REFRESH_TOKEN,
        "client_id": DROPBOX_CLIENT_ID,
        "client_secret": DROPBOX_CLIENT_SECRET
    }
    try:
        response = requests.post("https://api.dropbox.com/oauth2/token", data=payload, timeout=10)
        response.raise_for_status()
        token_data = response.json()
        return token_data["access_token"]
    except Exception as e:
        logger.error(f"❌ Failed to refresh Dropbox token: {e}")
        raise HTTPException(status_code=500, detail="Failed to refresh Dropbox token")

def get_ticket_thread(ticket_id):
    token = get_valid_access_token()
    headers = {"Authorization": f"Zoho-oauthtoken {token}"}
    url = f"{API_URL}/tickets/{ticket_id}/conversations"
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"❌ Error fetching ticket thread: {e}")
        return {}

def store_ticket_thread(ticket_id, thread_data):
    conn = sqlite3.connect(TEMP_LOCAL_DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ticket_threads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_id TEXT,
            sender TEXT,
            content TEXT,
            time TEXT,
            UNIQUE(ticket_id, time)
        )
    ''')

    values = []
    for item in thread_data.get("data", []):
        sender = item.get("fromEmail") or item.get("sender") or "unknown"
        content = html.unescape(item.get("content") or "").strip()
        timestamp = item.get("createdTime") or datetime.utcnow().isoformat()
        values.append((ticket_id, sender, content, timestamp))

    cursor.executemany(
        "INSERT OR IGNORE INTO ticket_threads (ticket_id, sender, content, time) VALUES (?, ?, ?, ?)",
        values
    )

    conn.commit()
    conn.close()

    try:
        access_token = get_dropbox_access_token()
        with open(TEMP_LOCAL_DB_PATH, "rb") as f:
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Dropbox-API-Arg": f'{{"path": "{DROPBOX_DB_PATH}","mode": "overwrite"}}',
                "Content-Type": "application/octet-stream"
            }
            response = requests.post("https://content.dropboxapi.com/2/files/upload", headers=headers, data=f)
            response.raise_for_status()
        logger.info("✅ Uploaded SQLite DB to Dropbox")
    except Exception as e:
        logger.error(f"❌ Dropbox upload failed: {e}")

@app.post("/update_ticket")
async def update_ticket(req: Request):
    try:
        body = await req.json()
        logger.info(f"🔍 REQUEST BODY: {body}")
        ticket_id = body.get("ticketId")
        if not ticket_id or not ticket_id.strip().isalnum():
            raise HTTPException(status_code=400, detail="Invalid ticketId format")

        thread = get_ticket_thread(ticket_id)
        if not thread.get("data"):
            raise HTTPException(status_code=404, detail="No conversations found for ticket")

        store_ticket_thread(ticket_id, thread)
        return {"status": "Ticket thread saved", "ticketId": ticket_id}

    except Exception as e:
        logger.error(f"❌ Exception in /update_ticket: {e}")
        raise HTTPException(status_code=500, detail="Failed in update_ticket")

@app.get("/inspect")
def inspect_ticket(ticket_id: str):
    try:
        conn = sqlite3.connect(TEMP_LOCAL_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT sender, content, time FROM ticket_threads WHERE ticket_id = ? ORDER BY time ASC", (ticket_id,))
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            raise HTTPException(status_code=404, detail="No data found for this ticket")

        return [{"sender": row[0], "content": row[1], "time": row[2]} for row in rows]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/answer")
def answer_ticket(req: AnswerRequest):
    try:
        conn = sqlite3.connect(TEMP_LOCAL_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT sender, content, time FROM ticket_threads WHERE ticket_id = ? ORDER BY time ASC", (req.ticketId,))
        rows = cursor.fetchall()
        conn.close()

        history = "\n".join([f"{row[0]}: {row[1]}" for row in rows])

        prompt = f"""
You are a helpful and precise dental support assistant.
You are responding on behalf of the same person who has replied earlier in this thread.

Previous conversation:
{history}

New customer message:
{req.question}

What is the best possible reply?
Please write only the message to the customer, not an explanation.
"""

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500
        )

        answer = response.choices[0].message.content.strip()
        return {"answer": answer, "ticketId": req.ticketId}

    except Exception as e:
        logger.error(f"❌ Exception in /answer: {e}")
        raise HTTPException(status_code=500, detail="Failed in /answer")

from webhook_integration import router as webhook_router
app.include_router(webhook_router)
