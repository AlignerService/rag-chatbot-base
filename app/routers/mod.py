# app/routers/mod.py
import aiosqlite
from datetime import datetime
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Request
from app.__init__ import DB_PATH, require_rag_token, md_to_plain

router = APIRouter(prefix="/mod", tags=["moderation"])

# Tabellen er allerede oprettet af migration 010_*.sql
# Kolonner antaget: id, created_at, status, session_id, ticket_id, contact_id,
# from_email, contact_name, subject, user_text, model_answer, editor_answer, approved_by

def _nowz() -> str:
    return datetime.utcnow().isoformat() + "Z"

@router.post("/intake", dependencies=[Depends(require_rag_token)])
async def intake(payload: Dict[str, Any]):
    # minimumfelter fra Wix intake
    session_id = (payload.get("sessionId") or "").strip()
    from_email = (payload.get("fromEmail") or "").strip().lower()
    contact_name = (payload.get("contactName") or "").strip()
    subject = (payload.get("subject") or "").strip()
    user_text = (payload.get("question") or "").strip()
    ticket_id = (payload.get("ticketId") or "").strip()
    contact_id = (payload.get("contactId") or "").strip()

    if not user_text:
        raise HTTPException(status_code=400, detail="question is required")

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT INTO moderation_queue
            (created_at, status, session_id, ticket_id, contact_id,
             from_email, contact_name, subject, user_text, model_answer, editor_answer)
            VALUES (?, 'pending', ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
        """, (_nowz(), session_id, ticket_id, contact_id, from_email, contact_name, subject, user_text))
        await db.commit()
        cur = await db.execute("SELECT last_insert_rowid()")
        rowid = (await cur.fetchone())[0]
    return {"ok": True, "id": rowid}

@router.get("/queue", dependencies=[Depends(require_rag_token)])
async def queue(status: str = "pending", limit: int = 50):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, created_at, status, from_email, contact_name, subject, user_text, model_answer, editor_answer "
            "FROM moderation_queue WHERE status=? ORDER BY id ASC LIMIT ?",
            (status, max(1, min(limit, 200)))
        ) as cur:
            rows = [dict(r) for r in await cur.fetchall()]
    return {"ok": True, "rows": rows}

@router.post("/save", dependencies=[Depends(require_rag_token)])
async def save(payload: Dict[str, Any]):
    id_ = payload.get("id")
    editor = (payload.get("editor_answer") or "").strip()
    if not id_:
        raise HTTPException(status_code=400, detail="id is required")
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE moderation_queue SET editor_answer=? WHERE id=?", (editor, id_)
        )
        await db.commit()
    return {"ok": True, "id": id_}

@router.post("/approve", dependencies=[Depends(require_rag_token)])
async def approve(payload: Dict[str, Any]):
    id_ = payload.get("id")
    who = (payload.get("approved_by") or "").strip() or "moderator"
    if not id_:
        raise HTTPException(status_code=400, detail="id is required")

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT user_text, editor_answer, model_answer FROM moderation_queue WHERE id=?", (id_,)
        ) as cur:
            row = await cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="not found")

        # prioritet: editor > model > tom
        final = (row["editor_answer"] or row["model_answer"] or "").strip()
        final = md_to_plain(final)  # sikker plain tekst til copy

        await db.execute(
            "UPDATE moderation_queue SET status='approved', approved_by=?, approved_at=? WHERE id=?",
            (who, _nowz(), id_)
        )
        await db.commit()

    return {"ok": True, "id": id_, "final": final}
