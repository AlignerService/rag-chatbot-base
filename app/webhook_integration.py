import os
import json
import aiohttp
import aiosqlite
from fastapi import APIRouter, HTTPException, Request
from datetime import datetime

# App-entry-point exports
from app import LOCAL_DB_PATH, sync_mgr

router = APIRouter()

ZOHO_TOKEN_URL = "https://accounts.zoho.eu/oauth/v2/token"
ZOHO_API_URL   = "https://desk.zoho.eu/api/v1"
TOKEN_CACHE    = os.getenv("ZOHO_TOKEN_CACHE", "token_cache.json")

# --- Token cache helpers ---
async def load_cached_token():
    try:
        with open(TOKEN_CACHE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("access_token") and data.get("expires_at", 0) > datetime.utcnow().timestamp():
            return data["access_token"]
    except Exception:
        pass
    return None

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
    with open(TOKEN_CACHE, "w", encoding="utf-8") as f:
        json.dump(token_data, f)
    return token_data["access_token"]

async def get_valid_zoho_token():
    token = await load_cached_token()
    if token:
        return token
    return await refresh_zoho_token()

# --- Webhook endpoint ---
@router.post("/webhook")
async def receive_ticket(req: Request):
    payload    = await req.json()
    ticket_id  = payload.get("ticketId")
    contact_id = payload.get("contactId")
    if not ticket_id or not contact_id:
        raise HTTPException(status_code=400, detail="Missing 'ticketId' or 'contactId'.")

    # Fetch Zoho conversations
    access_token = await get_valid_zoho_token()
    headers      = {"Authorization": f"Zoho-oauthtoken {access_token}"}
    async with aiohttp.ClientSession() as session:
        url = f"{ZOHO_API_URL}/tickets/{ticket_id}/conversations"
        async with session.get(url, headers=headers) as resp:
            resp.raise_for_status()
            data = await resp.json()
    threads = data.get("data", [])

    # Store to SQLite
    async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS ticket_threads (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id     TEXT,
                contact_id    TEXT,
                sender        TEXT,
                content       TEXT,
                created_time  TEXT,
                UNIQUE(ticket_id, created_time)
            )
        """)
        for t in threads:
            sender  = t.get("fromEmail") or t.get("sender") or "unknown"
            content = (t.get("content") or "").strip()
            created = t.get("createdTime") or datetime.utcnow().isoformat()
            await conn.execute(
                "INSERT OR IGNORE INTO ticket_threads "
                "(ticket_id, contact_id, sender, content, created_time) VALUES (?, ?, ?, ?, ?)",
                (ticket_id, contact_id, sender, content, created)
            )
        await conn.commit()

    # Trigger Dropbox sync
    await sync_mgr.queue()

    return {"status": "ok", "ticketId": ticket_id, "threads_stored": len(threads)}
