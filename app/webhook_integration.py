# webhook_integration.py (opdateret)
import os
import re
import json
import hmac
import html
import hashlib
import asyncio
from datetime import datetime, timezone

import aiohttp
import aiosqlite
from fastapi import APIRouter, HTTPException, Request

# App-entry-point exports
from app import LOCAL_DB_PATH, sync_mgr

router = APIRouter()

ZOHO_TOKEN_URL = "https://accounts.zoho.eu/oauth/v2/token"
ZOHO_API_URL   = "https://desk.zoho.eu/api/v1"

TOKEN_CACHE    = os.getenv("ZOHO_TOKEN_CACHE", "token_cache.json")
ZOHO_CLIENT_ID = os.getenv("ZOHO_CLIENT_ID", "")
ZOHO_CLIENT_SECRET = os.getenv("ZOHO_CLIENT_SECRET", "")
ZOHO_REFRESH_TOKEN  = os.getenv("ZOHO_REFRESH_TOKEN", "")

# -------------- helpers --------------

def _strip_html(text: str) -> str:
    if not text:
        return ""
    try:
        t = html.unescape(str(text))
    except Exception:
        t = str(text)
    # Fjern HTML tags
    t = re.sub(r"<[^>]+>", " ", t)
    # Trim lange citatblokke/signaturer (basic heuristik)
    t = re.sub(r"(?is)\n--+\s*[\s\S]{0,800}$", "", t)  # kut signatur
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _to_iso(dt) -> str:
    if not dt:
        return datetime.now(timezone.utc).isoformat()
    if isinstance(dt, (int, float)):
        return datetime.fromtimestamp(dt, tz=timezone.utc).isoformat()
    s = str(dt)
    # Zoho returnerer normalt ISO eller epoch ms; prøv at gætte
    try:
        # epoch ms
        if s.isdigit() and len(s) >= 12:
            return datetime.fromtimestamp(int(s)/1000.0, tz=timezone.utc).isoformat()
    except Exception:
        pass
    # fallback: lad den være
    return s

async def _ensure_schema(conn: aiosqlite.Connection):
    await conn.executescript("""
    PRAGMA journal_mode=WAL;
    PRAGMA synchronous=NORMAL;

    CREATE TABLE IF NOT EXISTS messages(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      ticketId     TEXT,
      messageId    TEXT,
      direction    TEXT,
      author       TEXT,
      author_email TEXT,
      createdAt    TEXT,
      language     TEXT,
      plainText    TEXT
    );

    /* Unik på Zoho's thread/message id hvis det findes */
    CREATE UNIQUE INDEX IF NOT EXISTS ux_messages_messageId
      ON messages(messageId) WHERE messageId IS NOT NULL AND messageId <> '';

    /* Fallback-idempotens hvis messageId ikke gives: */
    CREATE TABLE IF NOT EXISTS _msg_dedup (
      ticketId TEXT,
      createdAt TEXT,
      content_hash TEXT,
      UNIQUE(ticketId, createdAt, content_hash)
    );

    CREATE TABLE IF NOT EXISTS rag_chunks(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      text TEXT NOT NULL,
      source TEXT,
      ticketId TEXT,
      author TEXT,
      direction TEXT,
      createdAt TEXT
    );
    """)
    await conn.commit()

def _content_hash(txt: str) -> str:
    return hashlib.sha1(txt.encode("utf-8", errors="ignore")).hexdigest()

# -------------- OAuth --------------

async def _load_cached_token():
    try:
        with open(TOKEN_CACHE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("access_token") and data.get("expires_at", 0) > datetime.utcnow().timestamp():
            return data["access_token"]
    except Exception:
        pass
    return None

async def _refresh_zoho_token():
    if not (ZOHO_CLIENT_ID and ZOHO_CLIENT_SECRET and ZOHO_REFRESH_TOKEN):
        raise HTTPException(status_code=500, detail="Zoho OAuth creds mangler.")
    payload = {
        "refresh_token": ZOHO_REFRESH_TOKEN,
        "client_id":     ZOHO_CLIENT_ID,
        "client_secret": ZOHO_CLIENT_SECRET,
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

async def _get_valid_zoho_token():
    token = await _load_cached_token()
    if token:
        return token
    return await _refresh_zoho_token()

# -------------- Zoho fetch (med pagination) --------------

async def _fetch_all_conversations(ticket_id: str):
    """Returnerer liste af tråde for ticket_id. Håndterer pagination."""
    token   = await _get_valid_zoho_token()
    headers = {"Authorization": f"Zoho-oauthtoken {token}"}
    limit   = 100
    start   = 1
    out     = []

    async with aiohttp.ClientSession() as session:
        while True:
            url = f"{ZOHO_API_URL}/tickets/{ticket_id}/conversations?from={start}&limit={limit}"
            async with session.get(url, headers=headers) as resp:
                if resp.status == 404:
                    break
                resp.raise_for_status()
                data = await resp.json()

            rows = data.get("data") or data.get("conversations") or []
            if not rows:
                break
            out.extend(rows)

            info = data.get("info") or {}
            more = info.get("more") or info.get("has_more")
            if more:
                start = info.get("page", 1) * limit + 1 if "page" in info else (start + limit)
            else:
                break
    return out

# -------------- Webhook endpoint --------------

@router.post("/webhook")
async def receive_ticket(req: Request):
    """
    Forventet payload (fra din Zoho-automation/webhook):
      { "ticketId": "...", "contactId": "..." }
    Vi henter alle tråde for ticketId, og skriver dem i messages + rag_chunks.
    """
    body = await req.json()
    ticket_id  = str(body.get("ticketId") or "").strip()
    contact_id = str(body.get("contactId") or "").strip()
    if not ticket_id:
        raise HTTPException(status_code=400, detail="Missing 'ticketId'.")
    # contact_id bruges ikke direkte i indsættelsen, men beholdes for fremtidig berigelse

    # Hent Zoho-tråde
    try:
        threads = await _fetch_all_conversations(ticket_id)
    except aiohttp.ClientResponseError as e:
        raise HTTPException(status_code=502, detail=f"Zoho fetch failed: {e.status} {e.message}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Zoho fetch error: {e}")

    # Skriv til SQLite
    new_msg_ids = []
    async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
        await _ensure_schema(conn)

        # batch for messages
        msg_rows = []
        dedup_rows = []

        for t in threads:
            # Zoho feltnavne er ustabile. Vi favner bredt.
            message_id = str(t.get("id") or t.get("threadId") or t.get("messageId") or "").strip()
            author     = (t.get("fromEmail") or t.get("from") or t.get("sender") or t.get("senderEmail") or t.get("agentEmail") or "unknown")
            author     = str(author)
            direction  = (t.get("direction") or t.get("threadType") or t.get("type") or "")
            content    = (t.get("content") or t.get("threadContent") or t.get("description") or "").strip()
            created    = t.get("createdTime") or t.get("sentTime") or t.get("sentDate") or t.get("sentDateTime")
            language   = t.get("language") or None

            plain = _strip_html(content)
            if not plain:
                continue  # tomme tråde gider vi ikke

            created_iso = _to_iso(created)

            msg_rows.append((
                ticket_id,               # ticketId
                message_id,              # messageId
                str(direction or ""),    # direction
                author,                  # author
                None,                    # author_email (kan udfyldes senere)
                created_iso,             # createdAt
                language,                # language
                plain                    # plainText
            ))

            # dedup fallback (hvis no message_id)
            if not message_id:
                dedup_rows.append((ticket_id, created_iso, _content_hash(plain)))

        # idempotens: skriv fallback-unikke først
        if dedup_rows:
            await conn.executemany(
                "INSERT OR IGNORE INTO _msg_dedup(ticketId, createdAt, content_hash) VALUES (?,?,?)",
                dedup_rows
            )

        # indsæt messages med ON CONFLICT IGNORE via unique index på messageId
        await conn.executemany("""
            INSERT OR IGNORE INTO messages(ticketId,messageId,direction,author,author_email,createdAt,language,plainText)
            VALUES (?,?,?,?,?,?,?,?)
        """, msg_rows)

        # find hvilke der faktisk blev nye
        # strategi: slå op på (ticketId,messageId,createdAt,plainText) for nyligt hentede
        # for messageId=='' bruger vi dedup-tabellen
        new_ids = []
        for (ticketId, messageId, direction, author, author_email, createdAt, language, plainText) in msg_rows:
            if messageId:
                async with conn.execute(
                    "SELECT id FROM messages WHERE ticketId=? AND messageId=? LIMIT 1",
                    (ticketId, messageId)
                ) as cur:
                    r = await cur.fetchone()
                    if r:
                        new_ids.append(r[0])
            else:
                # slå op via dedup
                h = _content_hash(plainText)
                async with conn.execute(
                    "SELECT m.id FROM _msg_dedup d JOIN messages m ON m.ticketId=d.ticketId AND m.createdAt=d.createdAt WHERE d.ticketId=? AND d.createdAt=? AND d.content_hash=? ORDER BY m.id DESC LIMIT 1",
                    (ticketId, createdAt, h)
                ) as cur:
                    r = await cur.fetchone()
                    if r:
                        new_ids.append(r[0])

        # lav rag_chunks kun for de nye
        if new_ids:
            qmarks = ",".join("?" for _ in new_ids)
            await conn.execute(f"""
                INSERT INTO rag_chunks(text, source, ticketId, author, direction, createdAt)
                SELECT m.plainText, 'Zoho:webhook', m.ticketId, m.author, m.direction, m.createdAt
                FROM messages m
                WHERE m.id IN ({qmarks})
                  AND TRIM(COALESCE(m.plainText,'')) != ''
            """, new_ids)

        await conn.commit()

    # Sync til Dropbox (din egen manager)
    try:
        await sync_mgr.queue()
    except Exception:
        # ikke kritisk for HTTP-svaret
        pass

    return {
        "status": "ok",
        "ticketId": ticket_id,
        "received_threads": len(threads),
        "new_messages": len(new_msg_ids)  # bevidst "tynd"; vi indsatte via IGNORE
    }
