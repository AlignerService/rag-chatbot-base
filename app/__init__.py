# app/__init__.py
import os
import json
import logging
import asyncio
import hmac
import re
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np  # optional, used if you later expand retrieval
# Optional imports: we keep them but guard usage so the app still runs if missing
try:
    import faiss  # type: ignore
except Exception:
    faiss = None  # Safe fallback

try:
    import tiktoken  # type: ignore
except Exception:
    tiktoken = None  # Safe fallback

import aiohttp
import aiosqlite

from fastapi import FastAPI, HTTPException, Body, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# =========================
# Logging & Configuration
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("rag-app")

FAISS_INDEX_FILE  = os.getenv("FAISS_INDEX_FILE", "faiss.index")
METADATA_FILE     = os.getenv("METADATA_FILE", "metadata.json")
LOCAL_DB_PATH     = os.getenv("LOCAL_DB_PATH", "/mnt/data/rag.sqlite3")

# IMPORTANT: default to empty string, so comparisons are stable
RAG_BEARER_TOKEN  = os.getenv("RAG_BEARER_TOKEN", "")

OPENAI_MODEL      = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")

# Globals for initialized resources
_FAISS_INDEX = None
_METADATA: List[Dict[str, Any]] = []
_SQLITE_OK = False

# =========================
# FastAPI App & CORS
# =========================
app = FastAPI()

# Permissive for debugging; tighten (e.g. to ZoHo domains) in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Security (Bearer token)
# =========================
bearer_scheme = HTTPBearer(auto_error=True)

def require_rag_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    if credentials.scheme.lower() != "bearer":
        logger.info("Auth failed: missing Bearer scheme")
        raise HTTPException(status_code=403, detail="Invalid or missing RAG token")

    incoming = (credentials.credentials or "").strip()
    expected = (RAG_BEARER_TOKEN or "").strip()

    if not expected:
        logger.error("Auth failed: server RAG_BEARER_TOKEN not set")
        raise HTTPException(status_code=403, detail="Server token not configured")

    # Constant-time compare
    if not hmac.compare_digest(incoming, expected):
        logger.info("Auth failed: token mismatch")
        raise HTTPException(status_code=403, detail="Invalid or missing RAG token")

    return True

# =========================
# Startup / Shutdown
# =========================
@app.on_event("startup")
async def on_startup():
    if RAG_BEARER_TOKEN:
        logger.info("RAG_BEARER_TOKEN is set (value hidden)")
    else:
        logger.error("RAG_BEARER_TOKEN is missing!")

    # Ensure local dir for DB path (if applicable)
    try:
        base = os.path.dirname(LOCAL_DB_PATH)
        if base:
            os.makedirs(base, exist_ok=True)
    except Exception:
        pass

    try:
        await download_db()
        await init_db()
        logger.info("RAG startup complete")
    except Exception as e:
        logger.exception(f"Startup failed: {e}")

# =========================
# Utilities
# =========================
def detect_language(text: str) -> str:
    """
    Very simple heuristic: 'da' for Danish if signals present; otherwise 'en'.
    No external dependencies.
    """
    if not text:
        return "en"
    lowered = text.lower()

    # Special Danish chars
    if any(ch in lowered for ch in ["æ", "ø", "å"]):
        return "da"

    # Common Danish words/signals
    dk_signals = [
        " og ", " jeg ", " ikke", " det ", " der ", " som ", " har ", " skal ",
        " hvad", " hvordan", " hvorfor", " måske", " fordi", " gerne", " tak",
        " hej ", " kære "
    ]
    score = sum(1 for w in dk_signals if w in lowered)
    return "da" if score >= 2 else "en"


def make_system_prompt(lang: str) -> str:
    if lang == "da":
        return (
            "Du er en hjælpsom assistent for AlignerService. "
            "Svar præcist og kortfattet på dansk. "
            "Hvis du henviser til interne dokumenter, hold det neutralt og faktuelt."
        )
    else:
        return (
            "You are a helpful assistant for AlignerService. "
            "Answer precisely and concisely in English. "
            "If you refer to internal documents, keep it neutral and factual."
        )

# =========================
# Minimal RAG Core (safe fallbacks)
# Replace with your existing implementations if you have them.
# =========================

async def download_db():
    """
    OPTIONAL: Pull SQLite or index files from cloud storage.
    This function is a no-op by default, but kept async and logged.
    """
    logger.info("download_db(): no-op (override if you pull from cloud)")

async def init_db():
    """
    Initialize FAISS / load metadata / open SQLite.
    - Loads FAISS index if present
    - Loads metadata.json if present
    - Opens SQLite for future use (optional)
    """
    global _FAISS_INDEX, _METADATA, _SQLITE_OK

    # Load FAISS index if available
    if faiss is not None and os.path.exists(FAISS_INDEX_FILE):
        try:
            _FAISS_INDEX = faiss.read_index(FAISS_INDEX_FILE)
            logger.info(f"Loaded FAISS index from {FAISS_INDEX_FILE}")
        except Exception as e:
            logger.exception(f"Failed to load FAISS index: {e}")
            _FAISS_INDEX = None
    else:
        logger.info("FAISS not available or index file missing; continuing without FAISS.")

    # Load metadata.json if available
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                _METADATA = json.load(f)
            logger.info(f"Loaded metadata with {len(_METADATA)} entries from {METADATA_FILE}")
        except Exception as e:
            logger.exception(f"Failed to load metadata: {e}")
            _METADATA = []
    else:
        logger.info("No metadata.json found; continuing without metadata context.")

    # Try opening SQLite (optional)
    try:
        if os.path.exists(LOCAL_DB_PATH):
            async with aiosqlite.connect(LOCAL_DB_PATH) as db:
                await db.execute("SELECT 1")
            _SQLITE_OK = True
            logger.info(f"SQLite available at {LOCAL_DB_PATH}")
        else:
            _SQLITE_OK = False
            logger.info(f"SQLite path not found: {LOCAL_DB_PATH} (continuing)")
    except Exception as e:
        _SQLITE_OK = False
        logger.exception(f"SQLite check failed: {e}")

def _strip_html(text: str) -> str:
    """Meget enkel HTML-stripper, så keywords matcher bedre."""
    if not text:
        return ""
    # Fjern tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Komprimer whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _extract_text_from_meta(item: Any) -> str:
    """
    Ekstraher tekst robust fra forskellige metadata-formater.
    - Prøver en række gængse feltnavne
    - Leder også i simple nested dicts
    - Kan samle lister af strenge
    - Fallback: samler alle strengværdier i dict'en
    """
    if item is None:
        return ""

    # Direkte str
    if isinstance(item, str):
        return item

    # Liste af strenge eller mixed -> join
    if isinstance(item, list):
        parts = []
        for v in item:
            if isinstance(v, str):
                parts.append(v)
            elif isinstance(v, dict):
                nested = _extract_text_from_meta(v)
                if nested:
                    parts.append(nested)
        return "\n".join(parts).strip()

    # Dict: prøv prioriterede felter
    if isinstance(item, dict):
        candidates = [
            "text", "content", "chunk_text", "chunk", "page_text", "pageContent",
            "page_content", "body", "raw_text", "text_content", "md", "markdown",
            "html", "document_text", "passage", "excerpt", "summary", "data",
            "value", "message"
        ]
        for key in candidates:
            if key in item:
                val = item.get(key)
                if isinstance(val, str) and val.strip():
                    return val
                # Hvis feltet er nested dict/list – prøv at trække tekst ud
                if isinstance(val, (dict, list)):
                    nested = _extract_text_from_meta(val)
                    if nested:
                        return nested

        # Fallback: join ALLE streng-værdier i dict'en (kun top-level + simple nested)
        flat_strings = []
        for v in item.values():
            if isinstance(v, str):
                flat_strings.append(v)
            elif isinstance(v, (dict, list)):
                nested = _extract_text_from_meta(v)
                if nested:
                    flat_strings.append(nested)
        if flat_strings:
            return "\n".join(flat_strings).strip()

    return ""

def _keyword_score(text: str, query: str) -> int:
    """
    Very simple keyword overlap score. Replace with your vector similarity if needed.
    """
    if not text or not query:
        return 0
    t = text.lower()
    q_tokens = [tok for tok in query.lower().split() if len(tok) > 2]
    return sum(1 for tok in q_tokens if tok in t)

async def get_top_chunks(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Tolerant retrieval:
    - Prøver mange feltnavne i metadata for at finde tekst
    - Stripper simpel HTML
    - Returnerer top-k baseret på naiv keyword-score
    """
    if not _METADATA:
        logger.info("No metadata loaded; returning empty retrieval result.")
        return []

    q = (query or "").strip()
    scored = []

    for item in _METADATA:
        text = _extract_text_from_meta(item)
        text = _strip_html(text)
        if not text:
            continue

        score = _keyword_score(text, q)
        if score > 0:
            scored.append((score, text, item))

    # Hvis intet fik score > 0, returnér tomt (undgå at fylde modellen med irrelevant kontekst)
    if not scored:
        logger.info("No positive keyword matches in metadata for this query.")
        return []

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [{"text": s[1], "meta": s[2]} for s in scored[:k]]
    logger.info(f"Retrieved chunks: {len(top)} (from {len(_METADATA)} metadata items)")
    return top

# ------------ OpenAI call (async wrapper) ------------
async def get_rag_answer(final_prompt: str) -> str:
    """
    Calls OpenAI Chat Completions using either the modern client or legacy, whichever is available.
    Runs in a thread executor to avoid blocking the event loop if using sync clients.
    """
    if not OPENAI_API_KEY and not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY is not set; returning fallback answer.")
        return "I cannot reach the language model right now."

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _openai_complete_blocking, final_prompt)

def _openai_complete_blocking(final_prompt: str) -> str:
    # Try modern SDK first
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", OPENAI_API_KEY))
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": final_prompt},
            ],
            temperature=0.2,
        )
        content = resp.choices[0].message.content or ""
        return content.strip()
    except Exception as e_modern:
        logger.info(f"Modern OpenAI client not used ({e_modern}); trying legacy SDK...")

    # Fallback: legacy SDK
    try:
        import openai  # type: ignore
        openai.api_key = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY)
        resp = openai.ChatCompletion.create(
            model=os.getenv("OPENAI_MODEL", OPENAI_MODEL),
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": final_prompt},
            ],
            temperature=0.2,
        )
        content = resp["choices"][0]["message"]["content"] or ""
        return content.strip()
    except Exception as e_legacy:
        logger.exception(f"OpenAI call failed: {e_legacy}")
        return "I could not generate a response due to an internal error."

# =========================
# Endpoints
# =========================

@app.post("/debug-token")
async def debug_token(request: Request):
    auth_header = request.headers.get("authorization")
    return JSONResponse({"received_authorization": auth_header})

@app.post("/api/answer", dependencies=[Depends(require_rag_token)])
async def api_answer(request: Request):
    """
    Main endpoint called by ZoHo webhook / extension.
    Expects a JSON payload containing the email content somewhere.
    This implementation looks for common keys but falls back to raw JSON.
    """
    # 1) Parse body and log keys
    try:
        body = await request.json()
    except Exception as e:
        logger.exception(f"Invalid JSON body: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    try:
        logger.info(f"Incoming body keys: {list(body.keys())}")
    except Exception:
        pass

    # 2) Extract the text to feed into RAG
    #    Adapt these lines to fit your exact ZoHo payload.
    user_text = ""
    if isinstance(body, dict):
        # Try common fields first
        for key_guess in ["plainText", "text", "message", "content", "body"]:
            if key_guess in body and isinstance(body[key_guess], str):
                user_text = body[key_guess] or ""
                break
        if not user_text:
            # Nested path example: body["ticketThread"]["content"]
            thread = body.get("ticketThread") or {}
            if isinstance(thread, dict):
                for k in ["content", "plainText", "text"]:
                    if isinstance(thread.get(k), str) and thread.get(k):
                        user_text = thread[k]
                        break

    if not user_text:
        # Last resort: stringify the whole body
        user_text = json.dumps(body, ensure_ascii=False)

    user_text = user_text.strip()
    logger.info(f"user_text(sample 300): {user_text[:300]}")

    # 3) Detect language & create system prompt
    lang = detect_language(user_text)
    system_prompt = make_system_prompt(lang)

    # 4) Retrieval (ensure query is used!)
    try:
        top_chunks = await get_top_chunks(user_text)
    except Exception as e:
        logger.exception(f"get_top_chunks failed: {e}")
        top_chunks = []

    context = "\n\n".join(
        ch.get("text", "") if isinstance(ch, dict) else str(ch)
        for ch in top_chunks
    )[:8000]

    # 5) Compose final prompt (log a safe preview)
    final_prompt = (
        f"{system_prompt}\n\n"
        f"User message:\n{user_text}\n\n"
        f"Relevant context (may be partial):\n{context}\n\n"
        f"Answer:"
    )
    logger.info(f"final_prompt(sample 400): {final_prompt[:400]}")

    # 6) Call LLM
    try:
        answer = await get_rag_answer(final_prompt)
    except Exception as e:
        logger.exception(f"OpenAI call failed: {e}")
        answer = (
            "Beklager, der opstod en fejl under genereringen af svaret."
            if lang == "da" else
            "Sorry, an error occurred while generating the answer."
        )

    return {"finalAnswer": answer, "language": lang, "timestamp": datetime.utcnow().isoformat() + "Z"}
