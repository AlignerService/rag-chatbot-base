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

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)).strip())
    except Exception:
        return default

# Temperatur hentes fra Render env var (fallback 0.2)
CHAT_TEMPERATURE = _env_float("OPENAI_TEMPERATURE", 0.2)

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
# Svag-kontekst detektor (hallucinations-guard)
# ======================================================
def _weak_context(chunks, context_text: str, min_words: int = 220, min_sources: int = 2) -> bool:
    words = len((context_text or "").split())
    sources = len(chunks or [])
    return (words < min_words) or (sources < min_sources)


# ======================================================
# System prompts
# ======================================================
DEFAULT_SYSTEM_PROMPT = (
    "You are an experienced clinical advisor from AlignerService. "
    "You assist dentists and orthodontists with all aspects of clear aligner treatment — "
    "from clinical planning and troubleshooting to communication and practice optimization. "
    "Respond as a professional colleague, not as a salesperson. "
    "Use concise, practical language. Do not include greetings or signatures. "
    "Never include sections like 'What we need' or 'Next steps'. "
    "If key information is missing (e.g., overjet/overbite, molar/canine relation, crowding/spacing, arch coordination, Bolton), "
    "ask politely for exactly those within the answer. "
    "If the user mentions photos, radiographs/x-rays, CBCT, or intraoral scans, remind them to share such material ONLY "
    "via their own Doctor Platform (for GDPR/HIPAA compliance), never through this chat. "
    "Format:\n\n"
    "Brief conclusion:\n"
    "→ One or two sentences summarizing the key recommendation.\n\n"
    "Clinical guidance:\n"
    "1) Bullet list with essential clinical actions or considerations.\n"
    "2) Add brief rationale per point.\n\n"
    "If uncertain:\n"
    "→ State exactly which data would materially change the plan.\n\n"
    "Always answer in the user's language."
)

LOW_EVIDENCE_PROMPT = (
    "You are an experienced clinical advisor from AlignerService. "
    "EVIDENCE IS THIN: do not speculate or invent details. "
    "If it is not in the context, do not imply it. "
    "If the user mentions photos, radiographs/x-rays, CBCT, or intraoral scans, remind them to share such material ONLY "
    "via their own Doctor Platform (GDPR/HIPAA), never through this chat. "
    "Do not include greetings, signatures, or sections like 'What we need' or 'Next steps'. "
    "Format:\n\n"
    "Brief conclusion (provisional):\n"
    "→ 1–2 conservative sentences.\n\n"
    "Clinical guidance (basics only):\n"
    "• 3–5 safe actions/checks that hold without assumptions.\n\n"
    "Ask only for what is essential to proceed:\n"
    "• e.g., OJ, OB, molar/canine relation, crowding/spacing, arch coordination, Bolton; "
    "mention images/scans only if strictly necessary.\n\n"
    "Always answer in the user's language."
)


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

    # Retrieval-kontekst
    boosted_query = text
    qa_items = search_qa_json(boosted_query, k=30)
    qa_hits = [_qa_to_chunk(x) for x in qa_items]
    try:
        mao_kw = await get_mao_top_chunks(boosted_query, k=10)
    except Exception:
        mao_kw = []

    # Dedup de sammensatte kandidater
    cands = qa_hits + mao_kw
    uniq, seen = [], set()
    for c in cands:
        t = _strip_html(_extract_text_from_meta(c))
        if not t:
            continue
        k = t[:200].lower()
        if k in seen:
            continue
        uniq.append(c)
        seen.add(k)

    reranked = semantic_rerank(boosted_query, uniq, topk=min(10, len(uniq)))
    diversified = mmr_select(boosted_query, reranked, lam=0.4, m=min(8, len(reranked)))
    context = "\n\n".join(_strip_html(_extract_text_from_meta(x))[:1400] for x in diversified)[:7000]

    # Vælg prompt efter kontekststyrke
    system_prompt = LOW_EVIDENCE_PROMPT if _weak_context(diversified, context) else DEFAULT_SYSTEM_PROMPT

    # Stil-hint beholdes (kan hjælpe med markdown), men uden mail-fraser
    _ = style_hint("markdown", lang)

    final_prompt = (
        f"{system_prompt}\n\n"
        "CONTEXT (may be partial and noisy):\n"
        f"{context}\n\n"
        f"User:\n{text}\n\n"
        f"Answer for role {role_disp}. Keep to 6–10 short lines maximum. "
        "Focus on clinically actionable steps and only request data that truly changes the plan."
    )

    # Kald LLM med temperatur fra env; fallback til ældre signatur
    try:
        try:
            answer_md = await get_rag_answer(final_prompt, temperature=CHAT_TEMPERATURE)
        except TypeError:
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
