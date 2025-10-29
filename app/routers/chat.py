# app/routers/chat.py
import os
import hashlib
import hmac
import uuid
from typing import Optional
from datetime import datetime

import aiosqlite
from fastapi import APIRouter, HTTPException, Request

# Importer helpers fra app/__init__.py (samme som før)
from app import (
    logger, md_to_plain, get_rag_answer,
    detect_language, detect_role, role_label, detect_intent,
    style_hint,  # beholdes, men bruges kun let i prompten
    # Nedenstående beholdes i import for kompatibilitet, men bruges ikke i chat-svar
    stop_orders_block, customer_mail_guidance,
    search_qa_json, _qa_to_chunk, get_mao_top_chunks,
    semantic_rerank, mmr_select, _extract_text_from_meta, _strip_html
)

ENABLE = os.getenv("AI_CHAT_ENABLE", "1") == "1"
DB_PATH = os.getenv("LOCAL_DB_PATH", "/data/rag.sqlite3")

# ======================================================
# 1) Kolonne-map (peg kode mod dine rigtige felter)
# ======================================================
# chat schema mapping to existing DB
COL_SESSION_ID = "id"              # var 'session_id'
COL_EMAIL      = "contact_email"   # var 'user_email'
COL_STATUS     = "status"
COL_TOKEN_HASH = "token_hash"
COL_CREATED_AT = "created_at"

# FastAPI router
router = APIRouter(prefix="/chat", tags=["chat-public"])

# Hemmelighed til offentlig signatur
PUBLIC_SECRET = (os.getenv("RAG_PUBLIC_SECRET") or "").encode("utf-8")


# ======================================================
# Token utils (sign, verify, hash)
# ======================================================
def _sign_token(sid: str) -> str:
    if not PUBLIC_SECRET:
        raise HTTPException(status_code=500, detail="server misconfigured (no RAG_PUBLIC_SECRET)")
    sig = hmac.new(PUBLIC_SECRET, sid.encode("utf-8"), hashlib.sha256).hexdigest()[:32]
    return f"{sid}.{sig}"

def _verify_token(sid: str, token: str) -> bool:
    if not token or "." not in token or not PUBLIC_SECRET:
        return False
    try:
        t_sid, t_sig = token.split(".", 1)
    except ValueError:
        return False
    if t_sid != sid:
        return False
    expect = hmac.new(PUBLIC_SECRET, sid.encode("utf-8"), hashlib.sha256).hexdigest()[:32]
    return hmac.compare_digest(t_sig, expect)

def _hash_token(tok: str) -> str:
    return hashlib.sha256(tok.encode("utf-8")).hexdigest()


# ======================================================
# 2) Session-checker: brug id + kolonne-map
# ======================================================
async def _require_session(sid: str):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        sql = f"""
            SELECT {COL_SESSION_ID} AS sid, {COL_STATUS} AS status
            FROM chat_sessions
            WHERE {COL_SESSION_ID}=?
            LIMIT 1
        """
        async with db.execute(sql, (sid,)) as cur:
            row = await cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="session not found")
    if (row["status"] or "").lower() not in ("active", "open"):
        raise HTTPException(status_code=409, detail="session closed")


# ======================================================
# 3) /chat/start: indsæt i dine rigtige kolonner
# ======================================================
@router.post("/start")
async def start_session(payload: dict):
    if not ENABLE:
        raise HTTPException(status_code=503, detail="chat disabled")

    email = (payload.get("email") or "").strip().lower() if isinstance(payload, dict) else ""
    if not email:
        raise HTTPException(status_code=400, detail="missing email")
    if not PUBLIC_SECRET:
        raise HTTPException(status_code=500, detail="server misconfigured (no RAG_PUBLIC_SECRET)")

    sid   = uuid.uuid4().hex
    token = _sign_token(sid)
    now   = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            f"""
            INSERT INTO chat_sessions({COL_SESSION_ID}, {COL_EMAIL}, {COL_STATUS}, {COL_CREATED_AT}, {COL_TOKEN_HASH})
            VALUES(?, ?, 'active', ?, ?)
            """,
            (sid, email, now, _hash_token(token))
        )
        await db.commit()

    return {"sessionId": sid, "token": token}


# ======================================================
# 4) /chat/message: læs token fra header, match på id
#    Verificér signatur, tjek hash i DB, og kør RAG
# ======================================================
@router.post("/message")
async def chat_message(request: Request):
    if not ENABLE:
        raise HTTPException(status_code=503, detail="chat disabled")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid json")

    sid  = (body.get("sessionId") or "").strip()
    text = (body.get("text") or "").strip()
    if not sid or not text:
        raise HTTPException(status_code=400, detail="missing sessionId or text")

    token = request.headers.get("X-Chat-Token") or ""
    if not _verify_token(sid, token):
        raise HTTPException(status_code=401, detail="bad token")

    # tjek at session findes og er aktiv
    await _require_session(sid)

    # valider at token matcher hash i DB (ekstra sikkerhed)
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        sql = f"SELECT {COL_TOKEN_HASH} AS th FROM chat_sessions WHERE {COL_SESSION_ID}=? LIMIT 1"
        async with db.execute(sql, (sid,)) as cur:
            row = await cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="session not found")
    if row["th"] and row["th"] != _hash_token(token):
        raise HTTPException(status_code=401, detail="bad token")

    # ——— RAG forberedelse ———
    now  = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    lang = detect_language(text)
    role = detect_role(text, lang)
    role_disp = role_label(lang, role)
    intent = detect_intent(text, lang)

    # Gem user-besked
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO chat_messages(session_id, role, content, created_at) VALUES(?,?,?,?)",
            (sid, "user", text, now)
        )
        await db.commit()

    # === NY SYSTEM-PROMPT (engelsk, men instruerer om at svare på brugerens sprog) ===
    system_prompt = (
        "You are an experienced clinical advisor from AlignerService. "
        "You assist dentists and orthodontists with all aspects of clear aligner treatment — "
        "from clinical planning and troubleshooting to communication and practice optimization. "
        "Respond as a professional colleague, not as a salesperson. "
        "Use concise, practical language. Do not include greetings or signatures. "
        "Never add sections like 'What we need' or 'Next steps'. "
        "If key information is missing (e.g., OJ, OB, molar/canine relation, crowding/spacing, arch coordination, Bolton), "
        "ask for it politely within the answer. "
        "Use this structure:\n\n"
        "Brief conclusion:\n"
        "→ One or two sentences summarizing the key recommendation.\n\n"
        "Clinical guidance:\n"
        "1) Bullet list with the essential clinical actions or considerations.\n"
        "2) Briefly explain why each point matters.\n\n"
        "If uncertain:\n"
        "→ Mention exactly what data would help refine the plan.\n\n"
        "Always answer in the user's language."
    )

    # Retrival-kontekst
    boosted_query = text
    qa_items = search_qa_json(boosted_query, k=20)
    qa_hits = [_qa_to_chunk(x) for x in qa_items]
    try:
        mao_kw = await get_mao_top_chunks(boosted_query, k=8)
    except Exception:
        mao_kw = []

    # Dedup
    cands = qa_hits + mao_kw
    uniq, seen = [], set()
    for c in cands:
        t = _strip_html(_extract_text_from_meta(c))
        if not t:
            continue
        k = t[:160].lower()
        if k in seen:
            continue
        uniq.append(c)
        seen.add(k)

    reranked = semantic_rerank(boosted_query, uniq, topk=min(8, len(uniq)))
    diversified = mmr_select(boosted_query, reranked, lam=0.4, m=min(6, len(reranked)))

    context = "\n\n".join(_strip_html(_extract_text_from_meta(x))[:1200] for x in diversified)[:6000]

    # Stil-hint kan fortsat bruges, men vi undlader mail-guidance og stop-orders i chat
    style = style_hint("markdown", lang)
    # policy = stop_orders_block(lang)  # <- fjernet fra prompten for at undgå mail-fraser
    # if intent in ("status_request", "admin"):
    #     style += "\n" + customer_mail_guidance(lang, intent)

    final_prompt = (
        f"{system_prompt}\n\n"
        "CONTEXT (may be partial and noisy):\n"
        f"{context}\n\n"
        f"User:\n{text}\n\n"
        f"Answer for role {role_disp}. Keep to 6–10 short lines maximum. "
        "Focus on clinically actionable steps and only request data that truly changes the plan."
    )

    try:
        answer_md = await get_rag_answer(final_prompt)
    except Exception:
        logger.exception("chat LLM failed")
        answer_md = (
            "Beklager, der opstod en intern fejl. Prøv igen om et øjeblik."
            if lang == "da" else
            "Sorry, an internal error occurred. Please try again shortly."
        )

    # Gem assistant-besked
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO chat_messages(session_id, role, content, created_at) VALUES(?,?,?,?)",
            (sid, "assistant", answer_md, now)
        )
        await db.commit()

    return {
        "sessionId": sid,
        "reply_markdown": answer_md,
        "reply_plain": md_to_plain(answer_md),
        "language": lang
    }


# Historik: token-verificering + session-check
@router.get("/history/{session_id}")
async def chat_history(session_id: str, request: Request):
    token = request.headers.get("X-Chat-Token") or ""
    if not _verify_token(session_id, token):
        raise HTTPException(status_code=401, detail="bad token")

    await _require_session(session_id)

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT role, content, created_at FROM chat_messages WHERE session_id=? ORDER BY id ASC",
            (session_id,)
        ) as cur:
            rows = [dict(r) for r in await cur.fetchall()]

    return {"sessionId": session_id, "messages": rows}
