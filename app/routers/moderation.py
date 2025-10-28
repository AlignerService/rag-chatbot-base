# app/routers/moderation.py
import json
import aiosqlite
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any

from app.__init__ import DB_PATH, require_rag_token, get_rag_answer, md_to_plain  # reuse helpers

router = APIRouter(prefix="/mod", tags=["moderation"])

def _now():
    return datetime.utcnow().isoformat() + "Z"

async def _insert_draft(payload: Dict[str, Any]) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT INTO chat_drafts(
                created_at, status, session_id, ticket_id, contact_id,
                from_email, contact_name, subject, user_text, model_answer,
                editor_answer, send_channel
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            _now(), "pending",
            payload.get("sessionId"), payload.get("ticketId"), payload.get("contactId"),
            payload.get("fromEmail"), payload.get("contactName"), payload.get("subject"),
            payload.get("userText") or "",
            payload.get("modelAnswer") or None,
            None,
            payload.get("sendChannel") or "manual"
        ))
        await db.commit()
        cur = await db.execute("SELECT last_insert_rowid()")
        rid = (await cur.fetchone())[0]
        await cur.close()
        return int(rid)

@router.post("/intake", dependencies=[Depends(require_rag_token)])
async def intake(request: Request):
    """
    Modtager spørgsmål fra Wix, genererer MODEL-KLADDE med RAG og lægger i kø.
    Returnerer 202 + draft_id. Ingen svar sendes til kunden.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # træk felter
    user_text  = (body.get("question") or body.get("text") or "").strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Missing user text")
    subject    = (body.get("subject") or "").strip()
    from_email = (body.get("fromEmail") or "").strip().lower()
    session_id = (body.get("sessionId") or "").strip()
    ticket_id  = (body.get("ticketId") or "").strip()
    contact_id = (body.get("contactId") or "").strip()
    contact_nm = (body.get("contactName") or "").strip()

    # generér kort RAG-forslag i markdown og gem som model_answer
    try:
        prompt = f"User message:\n{user_text}\n\nReturn concise, clinically safe answer for a professional."
        ans_md = await get_rag_answer(prompt)
    except Exception:
        ans_md = "We will review your request and get back shortly."

    payload = {
        "sessionId": session_id,
        "ticketId": ticket_id,
        "contactId": contact_id,
        "fromEmail": from_email,
        "contactName": contact_nm,
        "subject": subject,
        "userText": user_text,
        "modelAnswer": ans_md,
        "sendChannel": "manual"
    }
    draft_id = await _insert_draft(payload)
    return JSONResponse({"ok": True, "draft_id": draft_id}, status_code=202)

@router.get("/queue", dependencies=[Depends(require_rag_token)])
async def queue(status: str = "pending", limit: int = 50):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        sql = """SELECT id, created_at, status, from_email, contact_name, subject,
                        substr(user_text,1,800) AS user_text,
                        model_answer, editor_answer
                 FROM chat_drafts
                 WHERE status IN (?, 'editing') 
                 ORDER BY id DESC LIMIT ?"""
        rows = [dict(r) for r in await (await db.execute(sql, (status, max(1, min(limit, 200))))).fetchall()]
    return {"ok": True, "rows": rows}

@router.post("/save", dependencies=[Depends(require_rag_token)])
async def save(request: Request):
    """Gem redigeret svar, sæt status=editing."""
    body = await request.json()
    draft_id = int(body.get("id") or 0)
    if not draft_id:
        raise HTTPException(400, "Missing id")
    editor_answer = (body.get("editor_answer") or "").strip()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""UPDATE chat_drafts
                            SET editor_answer=?, status='editing', updated_at=?
                            WHERE id=?""",
                         (editor_answer, _now(), draft_id))
        await db.commit()
    return {"ok": True, "id": draft_id}

@router.post("/approve", dependencies=[Depends(require_rag_token)])
async def approve(request: Request):
    """
    Markér som approved. Her kan du senere kalde ZoHo/SMTP.
    Returnerer den endelige tekst (editor > model).
    """
    body = await request.json()
    draft_id = int(body.get("id") or 0)
    approved_by = (body.get("approved_by") or "staff").strip()
    if not draft_id:
        raise HTTPException(400, "Missing id")

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        r = await (await db.execute("SELECT model_answer, editor_answer FROM chat_drafts WHERE id=?", (draft_id,))).fetchone()
        if not r:
            raise HTTPException(404, "Draft not found")
        final_text = (r["editor_answer"] or r["model_answer"] or "").strip()
        await db.execute("""UPDATE chat_drafts
                            SET status='approved', approved_by=?, approved_at=?, updated_at=?
                            WHERE id=?""",
                         (approved_by, _now(), _now(), draft_id))
        await db.commit()

    # her kunne du: send til ZoHo / email / webhook
    return {"ok": True, "id": draft_id, "final": md_to_plain(final_text)}

@router.post("/reject", dependencies=[Depends(require_rag_token)])
async def reject(request: Request):
    body = await request.json()
    draft_id = int(body.get("id") or 0)
    reason = (body.get("reason") or "").strip()
    if not draft_id:
        raise HTTPException(400, "Missing id")
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""UPDATE chat_drafts
                            SET status='rejected', updated_at=?, send_result=?
                            WHERE id=?""",
                         (_now(), reason[:500], draft_id))
        await db.commit()
    return {"ok": True, "id": draft_id}
