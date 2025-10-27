# app/routers/chat.py
import os, hashlib, hmac, uuid
from typing import Optional
from datetime import datetime

import aiosqlite
from fastapi import APIRouter, HTTPException, Header, Request

# Importer helpers fra app/__init__.py (samme som før)
from app import (
    logger, md_to_plain, get_rag_answer,
    detect_language, detect_role, role_label, detect_intent,
    style_hint, stop_orders_block, customer_mail_guidance,
    search_qa_json, _qa_to_chunk, get_mao_top_chunks,
    semantic_rerank, mmr_select, _extract_text_from_meta, _strip_html
)

ENABLE = os.getenv("AI_CHAT_ENABLE", "1") == "1"
DB_PATH = os.getenv("LOCAL_DB_PATH", "/data/rag.sqlite3")
PUBLIC_SECRET_ENV = os.getenv("RAG_PUBLIC_SECRET", "")

# FastAPI router
router = APIRouter(prefix="/chat", tags=["chat-public"])

def _hash_token(tok: str) -> str:
    return hashlib.sha256(tok.encode()).hexdigest()

def _require_secret_bytes() -> bytes:
    if not PUBLIC_SECRET_ENV:
        # 500 = serverkonfig fejl
        raise HTTPException(status_code=500, detail="server secret not set")
    return PUBLIC_SECRET_ENV.encode()

async def _require_session(token: Optional[str]) -> dict:
    """
    Validerer chat-token mod RAG_PUBLIC_SECRET og 003-chat_sessions
    (session_id TEXT PK, status TEXT='active' når session er åben)
    """
    if not token:
        raise HTTPException(status_code=401, detail="missing token")

    secret = _require_secret_bytes()
    # token format: "<session_id>.<sig>"
    try:
        sid, sig = token.split(".", 1)
    except ValueError:
        raise HTTPException(status_code=401, detail="bad token")

    good = hmac.new(secret, sid.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(good[:32], sig[:32]):  # kort sammenligning er nok
        raise HTTPException(status_code=401, detail="bad signature")

    # Verificér i DB ifølge 003-skemaet
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT session_id, status FROM chat_sessions WHERE session_id=? LIMIT 1",
            (sid,)
        ) as cur:
            row = await cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="session not found")
    if (row["status"] or "").lower() != "active":
        raise HTTPException(status_code=409, detail="session closed")

    return {"session_id": sid}

@router.post("/start")
async def start_session(request: Request):
    if not ENABLE:
        raise HTTPException(status_code=503, detail="chat disabled")

    # Body (valgfri)
    try:
        body = await request.json()
    except Exception:
        body = {}

    clinic_id = (body.get("clinicId") or "").strip() if isinstance(body, dict) else ""
    email     = (body.get("email") or "").strip().lower() if isinstance(body, dict) else ""

    # Token opbygning
    _ = _require_secret_bytes()  # fail fast hvis secret ikke sat
    sid = uuid.uuid4().hex
    sig = hmac.new(_.strip(), sid.encode(), hashlib.sha256).hexdigest()[:32]
    token = f"{sid}.{sig}"

    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    # 003: chat_sessions(session_id, user_email, status, created_at, updated_at, token_hash)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO chat_sessions(session_id, user_email, status, created_at, updated_at, token_hash)
            VALUES(?, ?, 'active', ?, ?, ?)
            """,
            (sid, email, now, now, _hash_token(token))
        )
        # Hvis du også vil gemme clinic_id et sted, kræver det en kolonne i 003.
        await db.commit()

    return {"sessionId": sid, "token": token}

@router.post("/message")
async def chat_message(request: Request, x_chat_token: Optional[str] = Header(default=None)):
    if not ENABLE:
        raise HTTPException(status_code=503, detail="chat disabled")

    sess = await _require_session(x_chat_token)

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid json")

    text = (body.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty text")

    sid = sess["session_id"]
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    # Gem user-besked (003: created_at NOT NULL)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO chat_messages(session_id, role, content, created_at) VALUES(?,?,?,?)",
            (sid, "user", text, now)
        )
        await db.commit()

    # Light RAG-svar
    lang = detect_language(text)
    role = detect_role(text, lang)
    role_disp = role_label(lang, role)
    intent = detect_intent(text, lang)

    system_prompt = (
        "Du svarer til en tandlæge/ortodontist via en offentlig chat. Svar kort, klart og med specifikke næste skridt. "
        "Spørg kun ind til det nødvendige for at kunne give et fagligt validt svar. "
        "Ingen patientnavne eller følsomme data."
        if lang == "da" else
        "You respond to a dentist/orthodontist via a public chat. Be concise, specific, and ask only for info needed."
    )

    # Retrieval: Q&A + MAO keyword
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
    style = style_hint("markdown", lang)
    policy = stop_orders_block(lang)
    if intent in ("status_request", "admin"):
        style += "\n" + customer_mail_guidance(lang, intent)

    final_prompt = (
        f"{system_prompt}\n\n"
        f"{policy}\n\n"
        "RELEVANT CONTEXT (may be partial):\n"
        f"{context}\n\n"
        f"User:\n{text}\n\n"
        f"Answer in {lang} for role {role_disp}. Keep to 6–10 korte linjer. "
        "Hvis der mangler nøgledata (Angle-klasse, OJ/OB, crowding/spacing mm, Bolton), bed om præcis det – intet udenom."
    )

    try:
        answer_md = await get_rag_answer(final_prompt)
    except Exception:
        logger.exception("chat LLM failed")
        answer_md = (
            "Beklager, der opstod en intern fejl. Prøv igen om et øjeblik."
            if lang == "da" else
            "Sorry, internal error."
        )

    # Gem assistant-besked (003: created_at NOT NULL)
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

@router.get("/history/{session_id}")
async def chat_history(session_id: str, x_chat_token: Optional[str] = Header(default=None)):
    # Kræv gyldig session/token
    await _require_session(x_chat_token)

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        # Stabil sortering efter id (003 har autoincrement)
        async with db.execute(
            "SELECT role, content, created_at FROM chat_messages WHERE session_id=? ORDER BY id ASC",
            (session_id,)
        ) as cur:
            rows = [dict(r) for r in await cur.fetchall()]

    return {"sessionId": session_id, "messages": rows}
