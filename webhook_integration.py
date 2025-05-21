from fastapi import APIRouter, HTTPException, Request
import requests
import sqlite3
import os
from datetime import datetime

router = APIRouter()

DATABASE_PATH = os.getenv("DATABASE_PATH", "/Users/macpro/Dropbox/AlignerService/RAG:Database:aktiv/rag.sqlite3")

@router.post("/webhook")
async def receive_ticket(req: Request):
    try:
        body = await req.json()
        ticket_id = body.get("ticketId")
        if not ticket_id:
            raise HTTPException(status_code=400, detail="No ticketId provided")

        access_token = get_valid_zoho_token()

        zoho_url = f"https://desk.zoho.eu/api/v1/tickets/{ticket_id}/threads"
        headers = {"Authorization": f"Zoho-oauthtoken {access_token}"}
        response = requests.get(zoho_url, headers=headers)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="ZoHo API failed")

        threads = response.json().get("data", [])
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ticket_threads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id TEXT,
                sender TEXT,
                content TEXT,
                created_time TEXT
            )
        ''')

        for thread in threads:
            cursor.execute("""
                INSERT INTO ticket_threads (ticket_id, sender, content, created_time)
                VALUES (?, ?, ?, ?)
            """, (
                ticket_id,
                thread.get("from", {}).get("email", "unknown"),
                thread.get("content", ""),
                thread.get("createdTime", datetime.utcnow().isoformat())
            ))

        conn.commit()
        conn.close()
        return {"status": "ok", "ticket_id": ticket_id, "threads_stored": len(threads)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def get_valid_zoho_token():
    import json

    token_file = "token_cache.json"
    with open(token_file, "r") as f:
        tokens = json.load(f)

    access_token = tokens.get("access_token")
    expires_at = tokens.get("expires_at")
    now = datetime.utcnow().timestamp()

    if access_token and expires_at and now < expires_at:
        return access_token

    client_id = os.getenv("ZOHO_CLIENT_ID")
    client_secret = os.getenv("ZOHO_CLIENT_SECRET")
    refresh_token = os.getenv("ZOHO_REFRESH_TOKEN")

    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token"
    }

    res = requests.post("https://accounts.zoho.eu/oauth/v2/token", params=payload)
    new_data = res.json()
    access_token = new_data["access_token"]
    expires_in = int(new_data["expires_in"])

    with open(token_file, "w") as f:
        json.dump({
            "access_token": access_token,
            "expires_at": datetime.utcnow().timestamp() + expires_in - 60
        }, f)

    return access_token
