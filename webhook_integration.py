# app/webhook_integration.py

from fastapi import APIRouter, HTTPException, Request
import os
import json
import asyncio
import aiosqlite
import aiohttp
from datetime import datetime
from dotenv import load_dotenv

# --- Load env & init Zoho token handling ---
load_dotenv()
router = APIRouter()

DATABASE_PATH  = os.getenv("LOCAL_DB_PATH", "/tmp/knowledge.sqlite")
ZOHO_TOKEN_URL = "https://accounts.zoho.eu/oauth/v2/token"
ZOHO_API_URL   = "https://desk.zoho.eu/api/v1"
TOKEN_CACHE    = os.getenv("ZOHO_TOKEN_CACHE", "token_cache.json")

# --- Helper to load cached token ---
async def load_cached_token():
    try:
        data = json.loads(open(TOKEN_CACHE).read())
        if data.get("access_token") and data.get("expires_at") > datetime.utcnow().timestamp():
            return data["access_token"]
    except Exception:
        pass
    return None

# --- Helper to refresh Zoho token ---
async def refresh_zoho_token():
    payload = {
        "refresh_token": os.getenv("ZOHO_REFRESH_TOKEN"),
        "client_id":     os.getenv("ZOHO_CLIENT_ID"),
        "client_secret": os.getenv("ZOHO_CLIENT_SECRET"),
        "grant_type":    "refresh_token"
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(ZOHO_TOKEN_URL, data=payload) as resp:
            resp.raise_for_status()
            token_data = await resp.json()
    token_data["expires_at"] = datetime.utcnow().timestamp() + token_data.get("expires_in", 0) - 60
    with open(TOKEN_CACHE, "w") as f:
        json.dump(token_data, f)
    return token_data["access_token"]

# --- Obtain a valid Zoho token ---
async def get_valid_zoho_token():
    token = await load_cached_token()
    if token:
        return token
    return await refresh_zoho_token()

@router.post("/webhook")
async def receive_ticket(req: Request):
    body = await req.json()
    ticket_id = body.get("ticketId")
    if not ticket_id:
        raise HTTPException(status_code=400, detail="No ticketId provided")

    # Fetch conversation threads
    access_token = await get_valid_zoho_token()
    headers = {"Authorization": f"Zoho-oauthtoken {access_token}"}
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{ZOHO_API_URL}/tickets/{ticket_id}/conversations", headers=headers) as resp:
            resp.raise_for_status()
            data = await resp.json()

    threads = data.get("data", [])

    # Store threads in SQLite
    async with aiosqlite.connect(DATABASE_PATH) as conn:
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS ticket_threads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id TEXT,
                sender TEXT,
                content TEXT,
                created_time TEXT
            )
        ''')
        for t in threads:
            sender  = t.get("fromEmail") or t.get("sender") or "unknown"
            content = t.get("content", "")
            created = t.get("createdTime") or datetime.utcnow().isoformat()
            await conn.execute(
                "INSERT OR IGNORE INTO ticket_threads (ticket_id, sender, content, created_time) VALUES (?, ?, ?, ?)",
                (ticket_id, sender, content, created)
            )
        await conn.commit()

    return {"status": "ok", "ticket_id": ticket_id, "threads_stored": len(threads)}
