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
# Optional imports: keep but guard usage so the app still runs if missing
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

from fastapi import FastAPI, HTTPException, Depends, Request
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
    Lightweight heuristic language detector for da/de/fr/en.
    Returns one of: 'da', 'de', 'fr', 'en'.
    """
    if not text:
        return "en"
    lowered = text.lower()

    # --- Danish ---
    if any(ch in lowered for ch in ["æ", "ø", "å"]):
        return "da"
    dk_signals = [
        " og ", " jeg ", " ikke", " det ", " der ", " som ", " har ", " skal ",
        " hvad", " hvordan", " hvorfor", " måske", " fordi", " gerne", " tak",
        " hej ", " kære "
    ]
    if sum(1 for w in dk_signals if w in lowered) >= 2:
        return "da"

    # --- German ---
    if any(ch in lowered for ch in ["ä", "ö", "ü", "ß"]):
        return "de"
    de_signals = [
        " und ", " nicht", " bitte", " danke", " hallo", " sie ", " ich ", " wir ",
        " zum ", " zur ", " mit ", " für ", " ueber", " über", " kein ", " keine ",
    ]
    if sum(1 for w in de_signals if w in lowered) >= 2:
        return "de"

    # --- French ---
    if any(ch in lowered for ch in ["à", "â", "æ", "ç", "é", "è", "ê", "ë", "î", "ï", "ô", "œ", "ù", "û", "ü", "ÿ"]):
        return "fr"
    fr_signals = [
        " bonjour", " merci", " s'il ", " s’", " vous ", " nous ", " je ", " tu ",
        " il ", " elle ", " avec ", " pour ", " mais ", " pas ", " est ", " aux ",
        " des ", " une ", " un "
    ]
    if sum(1 for w in fr_signals if w in lowered) >= 2:
        return "fr"

    # Default
    return "en"


def make_system_prompt(lang: str) -> str:
    if lang == "da":
        return (
            "Du er en hjælpsom assistent for AlignerService. "
            "Svar præcist og kortfattet på dansk. "
            "Konteksten nedenfor kan være på engelsk; det er helt i orden. "
            "Hvis du henviser til interne dokumenter, hold det neutralt og faktuelt."
        )
    if lang == "de":
        return (
            "Du bist eine hilfreiche Assistenz für AlignerService. "
            "Antworte präzise und knapp auf Deutsch. "
            "Der Kontext unten kann auf Englisch sein; das ist in Ordnung. "
            "Wenn du dich auf interne Dokumente beziehst, bleib neutral und sachlich."
        )
    if lang == "fr":
        return (
            "Vous êtes un assistant utile pour AlignerService. "
            "Répondez de manière précise et concise en français. "
            "Le contexte ci-dessous peut être en anglais ; c’est acceptable. "
            "Si vous faites référence à des documents internes, restez neutre et factuel."
        )
    # en
    return (
        "You are a helpful assistant for AlignerService. "
        "Answer precisely and concisely in English. "
        "The context below may be in English; that's fine. "
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
    """Simple HTML stripper to improve keyword matching."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)  # remove tags
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _extract_text_from_meta(item: Any) -> str:
    """
    Robust text extraction from various metadata formats.
    """
    if item is None:
        return ""

    if isinstance(item, str):
        return item

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
                if isinstance(val, (dict, list)):
                    nested = _extract_text_from_meta(val)
                    if nested:
                        return nested

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

# ------------ Translation helpers ------------
async def translate_to_english_if_needed(text: str, lang: str) -> str:
    """If input language is not English, translate the query to English for retrieval."""
    if lang == "en" or not text:
        return text
    prompt = (
        "Translate the following text to English only. "
        "Do not add explanations or metadata. Output only the translated text.\n\n"
        f"Text:\n{text}"
    )
    out = await _llm_short(prompt)
    return out.strip() if out else text

async def _llm_short(user_prompt: str) -> str:
    """Small helper to call the LLM with a short prompt (temperature 0)."""
    if not (OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")):
        logger.warning("OPENAI_API_KEY missing for translation; returning original text.")
        return ""

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _openai_complete_blocking_short, user_prompt)

def _openai_complete_blocking_short(user_prompt: str) -> str:
    # Try modern SDK first
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", OPENAI_API_KEY))
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You translate text precisely. Return only the translated text."},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e_modern:
        logger.info(f"Modern OpenAI client not used for short call ({e_modern}); trying legacy SDK...")

    try:
        import openai  # type: ignore
        openai.api_key = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY)
        resp = openai.ChatCompletion.create(
            model=os.getenv("OPENAI_MODEL", OPENAI_MODEL),
            messages=[
                {"role": "system", "content": "You translate text precisely. Return only the translated text."},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        return (resp["choices"][0]["message"]["content"] or "").strip()
    except Exception as e_legacy:
        logger.exception(f"Short OpenAI call failed: {e_legacy}")
        return ""

# ------------ Retrieval helpers (labels) ------------
def _label_from_meta(m):
    if isinstance(m, dict):
        return (
            m.get("title")
            or m.get("source")
            or m.get("path")
            or m.get("url")
            or m.get("id")
            or "metadata"
        )
    return "metadata"

# ------------ Retrieval ------------
async def get_top_chunks(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Tolerant retrieval based on metadata keyword overlap.
    Enforces a slightly stricter threshold (score >= 2).
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
        if score >= 2:  # stricter than >0
            scored.append((score, text, item))

    if not scored:
        logger.info("No positive keyword matches in metadata for this query.")
        return []

    scored.sort(key=lambda x: x[0], reverse=True)
    # preview top labels in logs
    try:
        preview = ", ".join(_label_from_meta(s[2]) for s in scored[:3])
        logger.info(f"Top source preview: {preview}")
    except Exception:
        pass

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

@app.get("/")
async def health():
    return {"status": "ok"}

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
    user_text = ""
    if isinstance(body, dict):
        for key_guess in ["plainText", "text", "message", "content", "body"]:
            if key_guess in body and isinstance(body[key_guess], str):
                user_text = body[key_guess] or ""
                break
        if not user_text:
            thread = body.get("ticketThread") or {}
            if isinstance(thread, dict):
                for k in ["content", "plainText", "text"]:
                    if isinstance(thread.get(k), str) and thread.get(k):
                        user_text = thread[k]
                        break

    if not user_text:
        user_text = json.dumps(body, ensure_ascii=False)

    user_text = user_text.strip()
    logger.info(f"user_text(sample 300): {user_text[:300]}")

    # 3) Detect language & create system prompt
    lang = detect_language(user_text)
    system_prompt = make_system_prompt(lang)

    # 4) Cross-lingual: translate query to English for retrieval if needed
    retrieval_query = await translate_to_english_if_needed(user_text, lang)
    if retrieval_query != user_text:
        logger.info("Query translated to English for retrieval.")
        logger.info(f"retrieval_query(sample 200): {retrieval_query[:200]}")

    # 5) Retrieval
    try:
        top_chunks = await get_top_chunks(retrieval_query)
    except Exception as e:
        logger.exception(f"get_top_chunks failed: {e}")
        top_chunks = []

    # === FAILSAFE GUARD: no context -> safe, generic answer without inventing details ===
    if not top_chunks:
        safe_generic = {
            "da": (
                "Jeg kan hjælpe med at strukturere jeres aligner-workflow i klare trin "
                "(screening, diagnose, planlægning, samtykke, opstart, kontroller, refinement, retention). "
                "Hvis du vil have et svar, der er direkte forankret i jeres egne materialer, skal jeg have adgang til RAG-kildedata."
            ),
            "de": (
                "Ich kann euren Aligner-Workflow in klaren Schritten strukturieren "
                "(Screening, Diagnose, Planung, Einverständnis, Start, Kontrollen, Refinement, Retention). "
                "Wenn die Antwort direkt auf euren eigenen Materialien basieren soll, brauche ich Zugriff auf die RAG-Quelldaten."
            ),
            "fr": (
                "Je peux structurer votre flux de travail aligneur en étapes claires "
                "(dépistage, diagnostic, planification, consentement, démarrage, contrôles, refinement, rétention). "
                "Si vous souhaitez une réponse basée directement sur vos propres ressources, j’ai besoin d’accéder aux données sources du RAG."
            ),
            "en": (
                "I can outline a clear, step-by-step aligner workflow "
                "(screening, diagnosis, planning, consent, start, reviews, refinement, retention). "
                "If you want an answer grounded in your own sources, I’ll need access to the RAG data."
            ),
        }
        # sources (likely empty) for UI consistency
        sources = []
        return {
            "finalAnswer": safe_generic.get(lang, safe_generic["en"]),
            "language": lang,
            "sources": sources,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

    context = "\n\n".join(
        ch.get("text", "") if isinstance(ch, dict) else str(ch)
        for ch in top_chunks
    )[:8000]

    # 6) Compose final prompt
    final_prompt = (
        f"{system_prompt}\n\n"
        f"IMPORTANT: Use only the information from 'Relevant context' below. "
        f"Do not invent names, case numbers or internal details that are not present in the context.\n\n"
        f"User message:\n{user_text}\n\n"
        f"Relevant context (may be in English and may be partial):\n{context}\n\n"
        f"Answer in the user's language (detected: {lang}):"
    )
    logger.info(f"final_prompt(sample 400): {final_prompt[:400]}")

    # 7) Call LLM
    try:
        answer = await get_rag_answer(final_prompt)
    except Exception as e:
        logger.exception(f"OpenAI call failed: {e}")
        answer = (
            "Beklager, der opstod en fejl under genereringen af svaret." if lang == "da"
            else "Es ist ein Fehler bei der Antwortgenerierung aufgetreten." if lang == "de"
            else "Désolé, une erreur s’est produite lors de la génération de la réponse." if lang == "fr"
            else "Sorry, an error occurred while generating the answer."
        )

    # 8) Build sources list for UI (max 3)
    sources = []
    for ch in top_chunks[:3]:
        meta = ch.get("meta", {})
        label = _label_from_meta(meta)
        url = meta.get("url") if isinstance(meta, dict) else None
        sources.append({"label": label, "url": url})

    return {
        "finalAnswer": answer,
        "language": lang,
        "sources": sources,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
