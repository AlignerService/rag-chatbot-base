# app/routers/mod.py
import os, json, aiosqlite, asyncio
from datetime import datetime
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException, Query
import httpx

router = APIRouter(prefix="/mod", tags=["moderation"])

# DB og intern base
DB_PATH = os.getenv("SQLITE_PATH", os.getenv("LOCAL_DB_PATH", "/data/rag.sqlite3"))
INTERNAL_BASE = os.getenv("INTERNAL_BASE", "http://127.0.0.1:10000")
RAG_BEARER_TOKEN = os.getenv("RAG_BEARER_TOKEN", "")

# simple helper
def _utcnow() -> str:
    return datetime.utcnow().isoformat() + "Z"

async def _ensure_table():
    # hvis din migration 010_create_moderation_table.sql allerede laver tabellen,
    # gør dette ingenting. Ellers oprettes en minimal tabel.
    sql = """
    CREATE TABLE IF NOT EXISTS moderation_queue(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        status TEXT NOT NULL,            -- pending | drafted | editing | approved
        session_id TEXT,
        ticket_id TEXT,
        contact_id TEXT,
        from_email TEXT,
        contact_name TEXT,
        subject TEXT,
        user_text TEXT NOT NULL,
        model_answer TEXT,               -- RAG kladde
        editor_answer TEXT,              -- redigeret af menneske
        final_answer TEXT,               -- det godkendte svar
        approved_by TEXT,
        approved_at TEXT
    );
    """
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(sql)
        await db.commit()

async def _rag_mail_draft(payload: Dict[str, Any]) -> str:
    """
    Kald din egen /api/answer for at få en mail-klar kladde.
    Vi bruger loopback-adressen i samme container for at undgå CORS.
    """
    if not RAG_BEARER_TOKEN:
        return ""  # ingen nøgle -> ingen draft, stadig ok
    body = {
        "output_mode": "mail",
        "fromEmail": payload.get("fromEmail") or payload.get("from_email") or "",
        "contactName": payload.get("contactName") or payload.get("contact_name") or "",
        "subject": payload.get("subject") or "",
        "question": payload.get("question") or payload.get("user_text") or "",
        "ticketId": payload.get("ticketId") or payload.get("ticket_id") or "",
        "contactId": payload.get("contactId") or payload.get("contact_id") or "",
    }
    headers = {"Authorization": f"Bearer {RAG_BEARER_TOKEN}", "Content-Type": "application/json"}
    url = f"{INTERNAL_BASE}/api/answer"
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(url, headers=headers, json=body)
            if r.status_code != 200:
                return ""
            data = r.json()
            # brug plain hvis muligt
            return (data.get("finalAnswerPlain")
                    or data.get("finalAnswer")
                    or data.get("message", {}).get("content")
                    or "")
    except Exception:
        return ""

def _row_to_dict(row: aiosqlite.Row) -> Dict[str, Any]:
    return {k: row[k] for k in row.keys()}

@router.post("/intake")
async def intake(payload: Dict[str, Any]):
    """
    Offentlig side rammer denne. Læg i kø og generér en model-kladde via /api/answer.
    Body felter vi bruger:
    sessionId, ticketId, contactId, fromEmail, contactName, subject, question
    """
    await _ensure_table()

    user_text = (payload.get("question") or "").strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="question is required")

    now = _utcnow()
    fields = {
        "session_id": (payload.get("sessionId") or "").strip(),
        "ticket_id": (payload.get("ticketId") or "").strip(),
        "contact_id": (payload.get("contactId") or "").strip(),
        "from_email": (payload.get("fromEmail") or "").strip().lower(),
        "contact_name": (payload.get("contactName") or "").strip(),
        "subject": (payload.get("subject") or "").strip(),
        "user_text": user_text,
    }

    # draft fra RAG
    draft = await _rag_mail_draft({
        "fromEmail": fields["from_email"],
        "contactName": fields["contact_name"],
        "subject": fields["subject"],
        "question": fields["user_text"],
        "ticketId": fields["ticket_id"],
        "contactId": fields["contact_id"],
    })

    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            """INSERT INTO moderation_queue
               (created_at, status, session_id, ticket_id, contact_id,
                from_email, contact_name, subject, user_text, model_answer)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (now, "drafted" if draft else "pending",
             fields["session_id"], fields["ticket_id"], fields["contact_id"],
             fields["from_email"], fields["contact_name"], fields["subject"],
             fields["user_text"], draft or None)
        )
        await db.commit()
        new_id = cur.lastrowid

    return {"ok": True, "id": new_id, "status": "drafted" if draft else "pending"}

@router.get("/queue")
async def queue(status: str = Query("pending"), limit: int = Query(50)):
    """Hent sager i køen (pending/drafted/editing)."""
    await _ensure_table()
    limit = max(1, min(limit, 200))
    valid = {"pending", "drafted", "editing", "approved"}
    if status not in valid:
        status = "pending"
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        sql = """SELECT id, created_at, status, from_email, contact_name, subject,
                        user_text, model_answer, editor_answer
                 FROM moderation_queue
                 WHERE status = ?
                 ORDER BY id DESC
                 LIMIT ?"""
        rows = []
        async with db.execute(sql, (status, limit)) as cur:
            async for r in cur:
                rows.append(_row_to_dict(r))
    return {"ok": True, "rows": rows}

@router.post("/save")
async def save(payload: Dict[str, Any]):
    """Gem kladde-tekst redigeret af medarbejder."""
    await _ensure_table()
    try:
        _id = int(payload.get("id"))
    except Exception:
        raise HTTPException(status_code=400, detail="id is required")
    editor = str(payload.get("editor_answer") or "").strip()
    if not editor:
        raise HTTPException(status_code=400, detail="editor_answer is required")

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE moderation_queue SET editor_answer=?, status='editing' WHERE id=?",
            (editor, _id)
        )
        await db.commit()
    return {"ok": True, "id": _id, "status": "editing"}

@router.post("/approve")
async def approve(payload: Dict[str, Any]):
    """Godkend svar. Vælger editor_answer hvis til stede, ellers model_answer."""
    await _ensure_table()
    try:
        _id = int(payload.get("id"))
    except Exception:
        raise HTTPException(status_code=400, detail="id is required")
    approved_by = (payload.get("approved_by") or "Staff").strip()

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT model_answer, editor_answer FROM moderation_queue WHERE id=?",
            (_id,)
        ) as cur:
            row = await cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="not found")
        final = (row["editor_answer"] or "").strip() or (row["model_answer"] or "").strip()
        if not final:
            raise HTTPException(status_code=400, detail="no draft to approve")

        await db.execute(
            "UPDATE moderation_queue SET status='approved', final_answer=?, approved_by=?, approved_at=? WHERE id=?",
            (final, approved_by, _utcnow(), _id)
        )
        await db.commit()

    return {"ok": True, "id": _id, "status": "approved", "final": final}
