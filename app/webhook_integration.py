# app/webhook_integration.py

import os
import json
import asyncio
from datetime import datetime

import aiohttp
import aiosqlite
from fastapi import APIRouter, HTTPException, Request

from app import LOCAL_DB_PATH, sync_mgr

router = APIRouter()

ZOHO_TOKEN_URL = "https://accounts.zoho.eu/oauth/v2/token"
ZOHO_API_URL   = "https://desk.zoho.eu/api/v1"
TOKEN_CACHE    = os.getenv("ZOHO_TOKEN_CACHE", "token_cache.json")


async def load_cached_token() -> str | None:
    """
    Returnér en gyldig access_token fra cache, hvis den ikke er udløbet.
    """
    try:
        data = json.loads(await asyncio.to_thread(open, TOKEN_CACHE, "r", encoding="utf-8").read())
        if data.get("access_token") and data.get("expires_at", 0) > datetime.utcnow().timestamp():
            return data["access_token"]
    except FileNotFoundError:
        return None
    except Exception:
        return None


async def refresh_zoho_token() -> str:
    """
    Hent nyt token fra Zoho OAuth og gem det i cache.
    """
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

    # Beregn udløbstid og gem i cache
    token_data["expires_at"] = datetime.utcnow().timestamp() + token_data.get("expires_in", 0) - 60
    await asyncio.to_thread(
        lambda d, fn: open(fn, "w", encoding="utf-8").write(json.dumps(d)),
        token_data, TOKEN_CACHE
    )
    return token_data["access_token"]


async def get_valid_zoho_token() -> str:
    """
    Returnér et gyldigt Zoho access_token, genudløs hvis nødvendigt.
    """
    token = await load_cached_token()
    if token:
        return token
    return await refresh_zoho_token()


@router.post("/webhook")
async def receive_ticket(req: Request):
    """
    Modtag webhook fra Zoho med ticketId + contactId,
    hent samtaler og gem i lokal SQLite via Dropbox‐DB.
    """
    payload    = await req.json()
    ticket_id  = payload.get("ticketId")
    contact_id = payload.get("contactId")
    if not ticket_id or not contact_id:
        raise HTTPException(status_code=400, detail="Missing 'ticketId' or 'contactId'.")

    # 1) Hent gyldigt Zoho-token
    access_token = await get_valid_zoho_token()
    headers      = {"Authorization": f"Zoho-oauthtoken {access_token}"}

    # 2) Hent samtale‐tråde fra Zoho
    async with aiohttp.ClientSession() as session:
        url = f"{ZOHO_API_URL}/tickets/{ticket_id}/conversations"
        async with session.get(url, headers=headers) as resp:
            resp.raise_for_status()
            data = await resp.json()
    threads = data.get("data", [])

    # 3) Gem i vores lokale SQLite
    async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
        # Sørg for tabellen eksisterer
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS ticket_threads (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id     TEXT,
                contact_id    TEXT,
                sender        TEXT,
                content       TEXT,
                created_time  TEXT,
                UNIQUE(ticket_id, created_time)
            );
        """)
        # Indsæt hver tråd, undgå duplikater
        for t in threads:
            sender  = t.get("fromEmail") or t.get("sender") or "unknown"
            content = (t.get("content") or "").strip()
            created = t.get("createdTime") or datetime.utcnow().isoformat()
            await conn.execute(
                "INSERT OR IGNORE INTO ticket_threads "
                "(ticket_id, contact_id, sender, content, created_time) "
                "VALUES (?, ?, ?, ?, ?)",
                (ticket_id, contact_id, sender, content, created)
            )
        await conn.commit()

    # 4) Trigger Dropbox‐sync
    await sync_mgr.queue()

    return {"status": "ok", "ticketId": ticket_id, "threads_stored": len(threads)}
