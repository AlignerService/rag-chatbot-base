# app/__init__.py
import os
import json
import logging
import asyncio
import hmac
import re
import hashlib
from collections import deque
from datetime import datetime
from typing import List, Dict, Any, Optional
from functools import lru_cache

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

try:
    import tiktoken  # type: ignore
except Exception:
    tiktoken = None

import aiohttp
import aiosqlite

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# =========================
# Logging & Config
# =========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("rag-app")

FAISS_INDEX_FILE   = os.getenv("FAISS_INDEX_FILE", "faiss.index")
METADATA_FILE      = os.getenv("METADATA_FILE", "metadata.json")
LOCAL_DB_PATH      = os.getenv("LOCAL_DB_PATH", "/mnt/data/knowledge.sqlite")

RAG_BEARER_TOKEN   = os.getenv("RAG_BEARER_TOKEN", "")

# Prefer OPENAI_MODEL; fall back to legacy OPENAI_CHAT_MODEL if present
OPENAI_MODEL       = os.getenv("OPENAI_MODEL") or os.getenv("OPENAI_CHAT_MODEL") or "gpt-4o-mini"
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "")

# -------- Dropbox creds (both modes supported) --------
DROPBOX_ACCESS_TOKEN  = os.getenv("DROPBOX_ACCESS_TOKEN", "")
DROPBOX_DB_PATH       = os.getenv("DROPBOX_DB_PATH", "")
DROPBOX_CLIENT_ID     = os.getenv("DROPBOX_CLIENT_ID", "")
DROPBOX_CLIENT_SECRET = os.getenv("DROPBOX_CLIENT_SECRET", "")
DROPBOX_REFRESH_TOKEN = os.getenv("DROPBOX_REFRESH_TOKEN", "")

# -------- Q&A JSON integration (env-config) --------
QA_JSON_PATH   = os.getenv("QA_JSON_PATH", "mao_qa_rag_export.json")
QA_JSON_ENABLE = os.getenv("QA_JSON_ENABLE", "1") == "1"

# Optional output mode: markdown | plain | tech_brief
OUTPUT_MODE_DEFAULT = os.getenv("OUTPUT_MODE", "markdown").lower()

# Globals
_FAISS_INDEX = None
_METADATA: List[Dict[str, Any]] = []
_SQLITE_OK = False

# Q&A globals
_QA_ITEMS: List[Dict[str, Any]] = []

# Anti-echo memory of last few answers (plain, clipped)
_LAST_HASHES: "deque[str]" = deque(maxlen=5)

# =========================
# FastAPI & CORS
# =========================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# =========================
# Security
# =========================
bearer_scheme = HTTPBearer(auto_error=True)
def require_rag_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    if credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=403, detail="Invalid or missing RAG token")
    incoming = (credentials.credentials or "").strip()
    expected = (RAG_BEARER_TOKEN or "").strip()
    if not expected:
        logger.error("Server RAG_BEARER_TOKEN not set")
        raise HTTPException(status_code=403, detail="Server token not configured")
    if not hmac.compare_digest(incoming, expected):
        raise HTTPException(status_code=403, detail="Invalid or missing RAG token")
    return True

# =========================
# Startup
# =========================
@app.on_event("startup")
async def on_startup():
    if RAG_BEARER_TOKEN:
        logger.info("RAG_BEARER_TOKEN is set (value hidden)")
    else:
        logger.error("RAG_BEARER_TOKEN is missing!")

    try:
        base = os.path.dirname(LOCAL_DB_PATH)
        if base:
            os.makedirs(base, exist_ok=True)
    except Exception:
        pass

    try:
        await download_db()
        await init_db()
        # --- load Q&A JSON (new) ---
        global _QA_ITEMS
        _QA_ITEMS = _qa_load_items()
        logger.info(f"Q&A loaded: {len(_QA_ITEMS)}")
        logger.info("RAG startup complete")
    except Exception as e:
        logger.exception(f"Startup failed: {e}")

# =========================
# Language detection
# =========================
def detect_language(text: str) -> str:
    if not text:
        return "en"
    lowered = text.lower()
    # da
    if any(ch in lowered for ch in ["æ", "ø", "å"]):
        return "da"
    if sum(1 for w in [" og ", " jeg ", " ikke", " det ", " der ", " som ", " tak", " klinik", "tandlæ", "ortodont"] if w in lowered) >= 2:
        return "da"
    # de
    if any(ch in lowered for ch in ["ä", "ö", "ü", "ß"]):
        return "de"
    if sum(1 for w in [" und ", " nicht", " bitte", " danke", " praxis", "zahn", "kfo", "empfang"] if w in lowered) >= 2:
        return "de"
    # fr
    if any(ch in lowered for ch in ["à", "â", "ç", "é", "è", "ê", "ë", "î", "ï", "ô", "œ", "ù", "û", "ü", "ÿ"]):
        return "fr"
    if sum(1 for w in [" bonjour", " merci", " cabinet", " clinique", " orthodont", " dentiste"] if w in lowered) >= 2:
        return "fr"
    return "en"

# =========================
# Role detection (heuristic)
# =========================
def detect_role(text: str, lang: str) -> str:
    t = (text or "").lower()
    lex = {
        "da": {
            "dentist": ["tandlæge", "tandlaege"],
            "orthodontist": ["ortodontist", "specialtandlæge i ortodonti", "kfo"],
            "assistant": ["klinikassistent", "assistent"],
            "hygienist": ["tandplejer", "dentalhygienist"],
            "receptionist": ["receptionist", "sekretær", "reception", "frontdesk"],
            "team": ["klinikteam", "team", "klinikpersonale"],
        },
        "en": {
            "dentist": ["dentist"],
            "orthodontist": ["orthodontist", "ortho"],
            "assistant": ["dental assistant", "assistant", "nurse", "dental nurse"],
            "hygienist": ["dental hygienist", "hygienist"],
            "receptionist": ["receptionist", "front desk", "practice manager"],
            "team": ["team", "practice team", "clinic team"],
        },
        "de": {}, "fr": {}
    }
    lang_map = lex.get(lang, lex["en"]) or lex["en"]
    scores = {r: 0 for r in ["dentist", "orthodontist", "assistant", "hygienist", "receptionist", "team"]}
    for role, keys in lang_map.items():
        for kw in keys:
            if kw in t:
                scores[role] += 1
    if all(v == 0 for v in scores.values()):
        admin_cues = ["book", "schedule", "reschedule", "appointment", "ombook", "termin", "rendez-vous", "accueil"]
        if any(c in t for c in admin_cues):
            return "receptionist"
        return "clinician"
    order = ["orthodontist", "dentist", "assistant", "hygienist", "receptionist", "team"]
    return max(order, key=lambda r: (scores[r], -order.index(r)))

def role_label(lang: str, role: str) -> str:
    labels = {
        "da": {"dentist": "tandlæge","orthodontist":"ortodontist","assistant":"klinikassistent","hygienist":"tandplejer","receptionist":"receptionist","team":"klinikteam","clinician":"kliniker"},
        "en": {"dentist": "dentist","orthodontist":"orthodontist","assistant":"dental assistant","hygienist":"dental hygienist","receptionist":"receptionist","team":"clinic team","clinician":"clinician"},
        "de": {"clinician":"Behandler/in"},
        "fr": {"clinician":"praticien(ne)"},
    }
    return labels.get(lang, labels["en"]).get(role, role)

# =========================
# Persona prompt (role-aware)
# =========================
def make_system_prompt(lang: str, role: str) -> str:
    role_text_da = {
        "dentist": "• Ret svar til tandlæger: kliniske parametre, planlægningsvalg, risici/kontraindikationer, dokumentation.",
        "orthodontist": "• Til ortodontister: staging, attachments/engagers, IPR-fordeling, elastikker, biomekanik.",
        "assistant": "• Til klinikassistenter: chairside-tjeklister, dokumentationsfelter, foto/scan-protokoller, eskalationskriterier.",
        "hygienist": "• Til tandplejere/hygienister: hygiejne/compliance, instruktion, observationer; ingen planændringer.",
        "receptionist": "• Til reception: skabeloner, (om)booking, forberedelsesliste; ingen kliniske råd.",
        "team": "• Til klinikteam: handoffs, checklister, opgavefordeling; kliniske beslutninger hos tandlæge/ortodontist.",
        "clinician": "• Til klinikere: fokus på kliniske parametre, protokoller, beslutningsregler.",
    }
    role_text_en = {
        "dentist": "• For dentists: clinical parameters, planning trade-offs, risks/contra-indications, documentation.",
        "orthodontist": "• For orthodontists: staging, attachments/engagers, IPR distribution, elastics, biomechanics.",
        "assistant": "• For dental assistants: chairside checklists, documentation fields, photo/scan protocols, escalation.",
        "hygienist": "• For hygienists: hygiene/compliance protocols, coaching, observations; no plan changes.",
        "receptionist": "• For reception: templates, (re)scheduling, pre-visit checklist; no clinical advice.",
        "team": "• For clinic teams: handoffs, checklists, task allocation; decisions stay with dentist/orthodontist.",
        "clinician": "• For clinicians: focus on clinical parameters, protocols, decision rules.",
    }
    role_map = {"da": role_text_da, "en": role_text_en}
    role_line = role_map.get(lang, role_text_en).get(role, role_map.get(lang, role_text_en)["clinician"])

    if lang == "da":
        return (
            "Du er AI-assistenten for tandlæge **Helle Hatt** (ekspert i clear aligners). "
            "Du svarer KUN til professionelle (tandlæger, ortodontister, klinikteams) — aldrig patienter.\n\n"
            "KILDER: Brug PRIMÆRT 'Relevant context' (Q&A, bog, historik). Hvis utilstrækkelig: sig det og giv kun etablerede best practices — "
            "opfind aldrig politikker, sagsnumre, navne eller data uden for konteksten.\n\n"
            f"ROLLEFOKUS\n{role_line}\n\n"
            "FORMAT\n• Kort konklusion (1–2 sætninger)\n• Struktureret protokol (nummererede trin med kliniske parametre: mm IPR, 22 t/d, staging)\n"
            "• Beslutningsregler (if/then) + risici/kontraindikationer\n• Næste skridt (2–4 punkter) + evt. journal-/opgavenote\n\n"
            "SIKKERHED\n• Ingen patient-specifik diagnose/ordination uden tilstrækkelig info; anfør usikkerheder kort. "
            "• Ingen direkte patienthenvendelse. Opret aldrig interne detaljer, der ikke er i konteksten."
        )
    return (
        "You are the AI assistant for **Dr. Helle Hatt** (clear-aligner expert). "
        "Address PROFESSIONALS ONLY.\n\n"
        "SOURCES: Rely on 'Relevant context'. If insufficient: state it and provide only established best practices — "
        "never invent policies, case numbers, names or data.\n\n"
        f"ROLE FOCUS\n{role_line}\n\n"
        "FORMAT\n• Brief takeaway (1–2 sentences)\n• Structured protocol (numbered; mm IPR, 22 h/day, staging)\n"
        "• Decision rules + risks/contra-indications\n• Next steps (2–4) + optional chart/task note\n\n"
        "SAFETY\n• No patient-specific diagnosis/prescription without adequate info; note uncertainties briefly. "
        "• Do not address patients. Never invent internal details not in context."
    )

# =========================
# Dropbox download (access token OR refresh flow)
# =========================
async def download_db():
    if not DROPBOX_DB_PATH:
        logger.info("download_db(): no-op (DROPBOX_DB_PATH not set)")
        return
    try:
        import dropbox  # type: ignore
    except Exception as e:
        logger.warning(f"Dropbox SDK not available ({e}); skipping DB download.")
        return

    try:
        if DROPBOX_ACCESS_TOKEN:
            dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)
        elif DROPBOX_REFRESH_TOKEN and DROPBOX_CLIENT_ID and DROPBOX_CLIENT_SECRET:
            dbx = dropbox.Dropbox(
                oauth2_refresh_token=DROPBOX_REFRESH_TOKEN,
                app_key=DROPBOX_CLIENT_ID,
                app_secret=DROPBOX_CLIENT_SECRET,
            )
        else:
            logger.warning("Dropbox credentials missing.")
            return

        dbx_path = DROPBOX_DB_PATH if DROPBOX_DB_PATH.startswith("/") else f"/{DROPBOX_DB_PATH}"
        local_dir = os.path.dirname(LOCAL_DB_PATH)
        if local_dir:
            os.makedirs(local_dir, exist_ok=True)
        dbx.files_download_to_file(LOCAL_DB_PATH, dbx_path)
        logger.info(f"Downloaded SQLite from Dropbox path '{dbx_path}' to {LOCAL_DB_PATH}")
    except Exception as e:
        logger.exception(f"Dropbox download failed: {e}")

# =========================
# Init FAISS / metadata / SQLite
# =========================
async def init_db():
    global _FAISS_INDEX, _METADATA, _SQLITE_OK

    if faiss is not None and os.path.exists(FAISS_INDEX_FILE):
        try:
            _FAISS_INDEX = faiss.read_index(FAISS_INDEX_FILE)
            logger.info(f"Loaded FAISS index from {FAISS_INDEX_FILE}")
        except Exception as e:
            logger.exception(f"Failed to load FAISS index: {e}")
            _FAISS_INDEX = None
    else:
        logger.info("FAISS not available or index missing; continuing without FAISS.")

    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                _METADATA = json.load(f)
            logger.info(f"Loaded metadata with {len(_METADATA)} entries from {METADATA_FILE}")
        except Exception as e:
            logger.exception(f"Failed to load metadata: {e}")
            _METADATA = []
    else:
        logger.info("No metadata.json found; continuing without metadata.")

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

# =========================
# SQLite helpers (Zoho ticket text)
# =========================
async def sqlite_latest_thread_plaintext(ticket_id: str, contact_id: Optional[str] = None) -> str:
    if not _SQLITE_OK:
        logger.warning("SQLite not available")
        return ""

    tid = str(ticket_id).strip()
    cid = str(contact_id).strip() if contact_id else None
    if not tid:
        return ""

    candidates_text = ["plainText", "plaintext", "text", "content", "body", "message", "message_text"]
    candidates_ticket = ["ticketId", "ticket_id", "tid", "ticket"]
    candidates_contact = ["contactId", "contact_id", "cid", "contact"]
    candidates_time = ["createdTime", "createdAt", "created_at", "date", "timestamp", "time", "updated_at"]

    try:
        async with aiosqlite.connect(LOCAL_DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            tables = []
            async with db.execute("SELECT name FROM sqlite_master WHERE type='table'") as cur:
                async for row in cur:
                    tables.append(row["name"])

            best_text = ""
            best_time = None

            for table in tables:
                async with db.execute(f"PRAGMA table_info('{table}')") as cur:
                    cols = [dict(row) async for row in cur]
                col_names = [c["name"] for c in cols]

                text_cols = [c for c in candidates_text if c in col_names]
                ticket_cols = [c for c in candidates_ticket if c in col_names]
                contact_cols = [c for c in candidates_contact if c in col_names]
                time_cols = [c for c in candidates_time if c in col_names]

                if not text_cols or not ticket_cols:
                    continue

                tcol = text_cols[0]
                kcol = ticket_cols[0]
                ccol = contact_cols[0] if contact_cols else None
                timecol = time_cols[0] if time_cols else None

                where = f"{kcol} = ?"
                params: List[Any] = [tid]
                if cid and ccol:
                    where += f" AND {ccol} = ?"
                    params.append(cid)

                order_clause = f"ORDER BY {timecol} DESC" if timecol else "ORDER BY rowid DESC"
                sql = f"SELECT {tcol} AS txt, {timecol if timecol else 'NULL'} AS t FROM '{table}' WHERE {where} {order_clause} LIMIT 1"

                try:
                    async with db.execute(sql, params) as cur:
                        row = await cur.fetchone()
                        if row:
                            txt = (row["txt"] or "").strip()
                            tval = row["t"]
                            if txt:
                                if best_time is None:
                                    best_text, best_time = txt, tval
                                else:
                                    if tval and (not best_time or str(tval) > str(best_time)):
                                        best_text, best_time = txt, tval
                                    elif not best_text:
                                        best_text = txt
                                        best_time = tval
                                logger.info(f"SQLite hit: table={table}, tcol={tcol}, kcol={kcol}, ccol={ccol}, time={timecol}")
                except Exception as e:
                    logger.info(f"SQLite query failed on {table}: {e}")
                    continue

            return best_text or ""
    except Exception as e:
        logger.exception(f"SQLite latest-thread lookup failed: {e}")
        return ""

# =========================
# Text helpers
# =========================
def _strip_html(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _extract_text_from_meta(item: Any) -> str:
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
    if not text or not query:
        return 0
    t = text.lower()
    q_tokens = [tok for tok in query.lower().split() if len(tok) > 2]
    return sum(1 for tok in q_tokens if tok in t)

def md_to_plain(md: str) -> str:
    if not md:
        return ""
    s = md
    s = re.sub(r"`{3}[\s\S]*?`{3}", "", s)
    s = re.sub(r"`([^`]+)`", r"\1", s)
    s = re.sub(r"\*\*(.*?)\*\*", r"\1", s)
    s = re.sub(r"_([^_]+)_", r"\1", s)
    s = re.sub(r"^#+\s*", "", s, flags=re.M)
    s = re.sub(r"^\s*[-*]\s+", "- ", s, flags=re.M)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def style_hint(mode: str, lang: str) -> str:
    mode = (mode or "markdown").lower()
    if lang == "da":
        if mode == "tech_brief":
            return (
                "SVARFORMAT (dansk): Returnér KUN en tekniker-klar receptblok i ren tekst, "
                "max 10 linjer, uden markdown.\n"
                "Brug labels og rækkefølgen her præcist:\n"
                "MÅL: <1 linje>\n"
                "INSTRUKTIONER: 1) <kort> 2) <kort> 3) <kort>\n"
                "VEDLÆG: <hvilke fotos/annoteringer>\n"
                "TJEK: <hvad skal verificeres i setup>\n"
                "Ingen forklaringer eller ekstra sektioner."
            )
        if mode == "plain":
            return "SVARFORMAT: Returnér samme indhold som normalt, men i ren tekst uden markdown, overskrifter eller emojis."
        return ""
    if mode == "tech_brief":
        return (
            "FORMAT: Return ONLY a technician-ready prescription block in plain text, "
            "max 10 lines, no markdown. Use exact labels:\n"
            "GOAL: <1 line>\n"
            "INSTRUCTIONS: 1) <short> 2) <short> 3) <short>\n"
            "ATTACH: <photos/annotations>\n"
            "CHECK: <what to verify in the setup>\n"
            "No extra sections."
        )
    if mode == "plain":
        return "FORMAT: Same content as usual, but plain text only. No markdown."
    return ""

# =========================
# LLM translation & calls
# =========================
async def translate_to_english_if_needed(text: str, lang: str) -> str:
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
    if not (OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")):
        logger.warning("OPENAI_API_KEY missing for translation")
        return ""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _openai_complete_blocking_short, user_prompt)

def _openai_complete_blocking_short(user_prompt: str) -> str:
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", OPENAI_API_KEY))
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You translate text precisely. Return only the translated text."},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e_modern:
        logger.info(f"Modern OpenAI short-call not used ({e_modern}); trying legacy...")
    try:
        import openai  # type: ignore
        openai.api_key = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY)
        resp = openai.ChatCompletion.create(
            model=os.getenv("OPENAI_MODEL", OPENAI_MODEL),
            messages=[
                {"role": "system", "content": "You translate text precisely. Return only the translated text."},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )
        return (resp["choices"][0]["message"]["content"] or "").strip()
    except Exception as e_legacy:
        logger.exception(f"Short OpenAI call failed: {e_legacy}")
        return ""

async def get_rag_answer(final_prompt: str) -> str:
    if not OPENAI_API_KEY and not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY missing; returning fallback")
        return "I cannot reach the language model right now."
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _openai_complete_blocking, final_prompt)

def _openai_complete_blocking(final_prompt: str) -> str:
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", OPENAI_API_KEY))
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": final_prompt},
            ],
            temperature=0.1,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e_modern:
        logger.info(f"Modern OpenAI client not used ({e_modern}); trying legacy...")
    try:
        import openai  # type: ignore
        openai.api_key = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY)
        resp = openai.ChatCompletion.create(
            model=os.getenv("OPENAI_MODEL", OPENAI_MODEL),
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": final_prompt},
            ],
            temperature=0.1,
        )
        return (resp["choices"][0]["message"]["content"] or "").strip()
    except Exception as e_legacy:
        logger.exception(f"OpenAI call failed: {e_legacy}")
        return "I could not generate a response due to an internal error."

# =========================
# Retrieval helpers (generic metadata)
# =========================
def _label_from_meta(m):
    if isinstance(m, dict):
        return m.get("title") or m.get("source") or m.get("path") or m.get("url") or m.get("id") or "metadata"
    return "metadata"

async def get_top_chunks(query: str, k: int = 5) -> List[Dict[str, Any]]:
    if not _METADATA:
        logger.info("No metadata loaded; returning empty retrieval result.")
        return []
    q = (query or "").strip()
    scored = []
    for item in _METADATA:
        text = _strip_html(_extract_text_from_meta(item))
        if not text:
            continue
        score = _keyword_score(text, q)
        if score >= 2:
            scored.append((score, text, item))
    if not scored:
        logger.info("No positive keyword matches in metadata for this query.")
        return []
    scored.sort(key=lambda x: x[0], reverse=True)
    try:
        preview = ", ".join(_label_from_meta(s[2]) for s in scored[:3])
        logger.info(f"Top source preview: {preview}")
    except Exception:
        pass
    top = [{"text": s[1], "meta": s[2]} for s in scored[:k]]
    logger.info(f"Retrieved chunks: {len(top)} (from {len(_METADATA)} metadata items)")
    return top

# =========================
# Q&A JSON search (NEW)
# =========================
def _qa_log(msg: str):
    try:
        logger.info(f"[Q&A JSON] {msg}")
    except Exception:
        print(f"[Q&A JSON] {msg}")

def _qa_load_items() -> List[Dict[str, Any]]:
    if not QA_JSON_ENABLE:
        _qa_log("disabled via QA_JSON_ENABLE")
        return []
    try:
        with open(QA_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            _qa_log("unexpected JSON format (not a list) – disabling")
            return []
        _qa_log(f"loaded {len(data)} Q&A items from {QA_JSON_PATH}")
        return data
    except FileNotFoundError:
        _qa_log(f"file not found: {QA_JSON_PATH} – disabling")
    except Exception as e:
        _qa_log(f"load error: {e} – disabling")
    return []

def _qa_tokenize(s: str) -> List[str]:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9æøåéüöß\s\-]", " ", s)
    return [t for t in s.split() if len(t) >= 2]

_QA_WEIGHTS = {
    "refinement": 3.0, "refinements": 3.0, "tracking": 2.0, "elastic": 2.0, "elastics": 2.0,
    "class": 0.5, "ii": 0.8, "iii": 0.8, "attachment": 2.0, "attachments": 2.0, "ipr": 2.0,
    "anchorage": 2.0, "torque": 2.0, "intrusion": 1.8, "extrusion": 1.8, "rotation": 1.2,
    "crossbite": 1.8, "retention": 1.8, "scan": 1.2, "x-rays": 1.2, "cbct": 1.2, "trimline": 1.5,
}

def _qa_score(query_tokens: List[str], haystack_tokens: List[str]) -> float:
    if not haystack_tokens:
        return 0.0
    hs = set(haystack_tokens)
    score = 0.0
    for t in set(query_tokens):
        if t in hs:
            score += 1.0 + _QA_WEIGHTS.get(t, 0.0)
    return score

@lru_cache(maxsize=512)
def _qa_cached(norm_query: str, k: int) -> tuple:
    if not _QA_ITEMS:
        return tuple()
    qtok = norm_query.split()
    scored = []
    for it in _QA_ITEMS:
        hay = " ".join([it.get("question", "")] + (it.get("synonyms", []) or []) + [it.get("answer_markdown", "")])
        stokens = _qa_tokenize(hay)
        s = _qa_score(qtok, stokens)
        if s > 0:
            scored.append((s, it))
    scored.sort(key=lambda x: x[0], reverse=True)
    light = tuple(json.dumps({"id": it.get("id"), "question": it.get("question")}, ensure_ascii=False) for _, it in scored[:k])
    return light

def search_qa_json(query: str, k: int = 5) -> List[Dict[str, Any]]:
    if not _QA_ITEMS or not query:
        return []
    norm = " ".join(_qa_tokenize(query))
    light = _qa_cached(norm, k)
    if not light:
        return []
    ids = {json.loads(s)["id"] for s in light}
    return [it for it in _QA_ITEMS if it.get("id") in ids]

def _qa_to_chunk(it: Dict[str, Any]) -> Dict[str, Any]:
    text = f"Q: {it.get('question','')}\n\n{it.get('answer_markdown','')}"
    meta = {
        "source": "Q&A",
        "id": it.get("id"),
        "title": it.get("question"),
        "refs": it.get("refs", []),
        "category": it.get("category"),
        "tags": it.get("tags", []),
    }
    return {"text": text, "meta": meta}

# =========================
# MAO via anchors (prefer from METADATA)
# =========================
def _get_meta_value(item: Dict[str, Any], key: str):
    if not isinstance(item, dict):
        return None
    if key in item:
        return item.get(key)
    m = item.get("meta")
    if isinstance(m, dict) and key in m:
        return m.get(key)
    return None

def _is_mao_item(item: Dict[str, Any]) -> bool:
    src = (_get_meta_value(item, "source") or _get_meta_value(item, "SOURCE") or "").upper()
    return "MAO" in src or src == "BOOK" or "MASTERING" in src

def find_mao_by_anchors(anchor_ids: List[str], k_per_anchor: int = 2) -> List[Dict[str, Any]]:
    if not _METADATA or not anchor_ids:
        return []
    want = set(anchor_ids)
    hits: List[Dict[str, Any]] = []
    for it in _METADATA:
        if not _is_mao_item(it):
            continue
        a_id = _get_meta_value(it, "anchor_id") or _get_meta_value(it, "anchorId")
        if a_id and a_id in want:
            txt = _extract_text_from_meta(it)
            if txt:
                hits.append({"text": txt, "meta": it})
    out: List[Dict[str, Any]] = []
    per = {aid: 0 for aid in anchor_ids}
    for h in hits:
        aid = _get_meta_value(h["meta"], "anchor_id") or _get_meta_value(h["meta"], "anchorId")
        if aid in per and per[aid] < k_per_anchor:
            out.append(h); per[aid] += 1
    return out

async def get_mao_top_chunks(query: str, k: int = 4) -> List[Dict[str, Any]]:
    if not _METADATA:
        return []
    q = (query or "").strip()
    scored = []
    for item in _METADATA:
        if not _is_mao_item(item):
            continue
        text = _strip_html(_extract_text_from_meta(item))
        if not text:
            continue
        score = _keyword_score(text, q)
        if score >= 2:
            scored.append((score, text, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"text": s[1], "meta": s[2]} for s in scored[:k]]

# =========================
# Endpoints
# =========================
@app.get("/")
async def health():
    return {"status": "ok", "qa_loaded": len(_QA_ITEMS), "metadata_loaded": len(_METADATA)}

@app.post("/debug-token")
async def debug_token(request: Request):
    return JSONResponse({"received_authorization": request.headers.get("authorization")})

@app.post("/api/answer", dependencies=[Depends(require_rag_token)])
async def api_answer(request: Request):
    # 1) Parse body
    try:
        body = await request.json()
    except Exception as e:
        logger.exception(f"Invalid JSON body: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    logger.info(f"Incoming body keys: {list(body.keys()) if isinstance(body, dict) else type(body)}")

    # Output mode
    output_mode = OUTPUT_MODE_DEFAULT
    if isinstance(body, dict) and "output_mode" in body:
        output_mode = str(body.get("output_mode") or OUTPUT_MODE_DEFAULT).lower()

    # 2) Extract Zoho-style IDs first (preferred flow)
    user_text = ""
    ticket_id = None
    contact_id = None

    if isinstance(body, dict):
        if "ticketId" in body and "question" in body:
            ticket_id = str(body.get("ticketId") or "").strip()
            contact_id = str(body.get("contactId") or "").strip() if body.get("contactId") else None
            txt = await sqlite_latest_thread_plaintext(ticket_id, contact_id)
            if txt:
                user_text = txt
                logger.info("Zoho ID-only: pulled latest thread from SQLite.")
            else:
                user_text = str(body.get("question") or "").strip()
                logger.info("Zoho ID-only: SQLite empty; using 'question' field.")
        if not user_text:
            for key_guess in ["plainText", "text", "message", "content", "body", "question"]:
                if key_guess in body and isinstance(body[key_guess], str) and body[key_guess].strip():
                    user_text = body[key_guess].strip()
                    break
            if not user_text:
                thread = body.get("ticketThread") or {}
                if isinstance(thread, dict):
                    for k in ["content", "plainText", "text"]:
                        if isinstance(thread.get(k), str) and thread.get(k):
                            user_text = thread[k].strip()
                            break

    if not user_text:
        user_text = json.dumps(body, ensure_ascii=False)

    logger.info(f"user_text(sample 300): {user_text[:300]}")

    # 3) Language + Role + Prompt
    lang = detect_language(user_text)
    role = detect_role(user_text, lang)
    role_disp = role_label(lang, role)
    system_prompt = make_system_prompt(lang, role)

    # 4) Cross-lingual retrieval
    retrieval_query = await translate_to_english_if_needed(user_text, lang)
    if retrieval_query != user_text:
        logger.info("Query translated to English for retrieval.")
        logger.info(f"retrieval_query(sample 200): {retrieval_query[:200]}")

    # --- Q&A hits (JSON, prioritized) ---
    qa_items = search_qa_json(retrieval_query, k=3)
    qa_hits = [_qa_to_chunk(x) for x in qa_items]

    # --- MAO via anchors from Q&A refs (force include if available) ---
    qa_refs = [aid for it in qa_items for aid in (it.get("refs") or []) if isinstance(aid, str) and aid.startswith("MAO:")]
    mao_ref_hits = find_mao_by_anchors(qa_refs, k_per_anchor=1) if qa_refs else []

    # --- metadata fallback (MAO keyword) ---
    mao_kw_hits = []
    if not mao_ref_hits:
        try:
            mao_kw_hits = await get_mao_top_chunks(retrieval_query, k=3)
        except Exception as e:
            logger.info(f"MAO keyword retrieval failed: {e}")

    # --- historical tone (last thread), optional ---
    history_hits = []
    if ticket_id:
        hist = await sqlite_latest_thread_plaintext(ticket_id, contact_id)
        if hist:
            history_hits = [{"text": hist[:1500], "meta": {"source": "History", "title": "Recent thread"}}]

    # Merge in fixed order; always include up to 2 MAO refs when present
    combined_chunks = qa_hits[:3] + mao_ref_hits[:2] + mao_kw_hits[:2] + history_hits[:1]

    # 5) Failsafe if no context at all
    if not combined_chunks:
        safe_generic = {
            "da": (
                "Grundprotokol når kontekst mangler:\n"
                "1) Screening/diagnose → 2) Plan (staging, IPR, attachments) → 3) Start → 4) Kontroller/tracking → 5) Refinement → 6) Retention.\n"
                "Næste skridt: indlæs flere kildedata i RAG for mere målrettet svar."
            ),
            "en": (
                "Baseline protocol when context is missing:\n"
                "1) Screening/diagnosis → 2) Planning (staging, IPR, attachments) → 3) Start → 4) Reviews/tracking → 5) Refinement → 6) Retention.\n"
                "Next steps: load more sources into the RAG."
            ),
        }
        sources = []
        answer = safe_generic.get(lang, safe_generic["en"])
        return {
            "finalAnswer": answer,
            "finalAnswerMarkdown": answer,
            "finalAnswerPlain": md_to_plain(answer),
            "language": lang,
            "role": role_disp,
            "sources": sources,
            "ticketId": ticket_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "message": {"content": answer, "language": lang}
        }

    # Context text
    def _clip(txt: str, max_len: int = 1200) -> str:
        return (txt[:max_len] + "…") if len(txt) > max_len else txt

    context = "\n\n".join(_clip(ch.get("text", "") if isinstance(ch, dict) else str(ch)) for ch in combined_chunks)[:8000]

    # 6) Compose final prompt (with strict policies)
    style = style_hint(output_mode, lang)
    language_policy = f"LANGUAGE POLICY: Respond strictly in {lang}. If context is in another language, translate clinically and naturally to {lang}."
    ranked_policy = (
        "RANKED CONTEXT POLICY\n"
        "• Q&A evidence = primary clinical guidance.\n"
        "• MAO (by anchors) = authoritative. If conflict, prefer MAO.\n"
        "• MAO (keyword) = secondary fallback.\n"
        "• History = tone only; do not override facts.\n"
        "• If evidence is weak or absent: explicitly state 'insufficient context' and DO NOT invent numeric parameters (mm IPR, degrees, hours/day, forces).\n"
    )

    final_prompt = (
        f"{system_prompt}\n\n"
        f"{ranked_policy}"
        f"{style}\n"
        f"{language_policy}\n\n"
        f"IMPORTANT: Use only the information from 'Relevant context' below. "
        f"Do not invent names, case numbers or internal details that are not present in the context.\n\n"
        f"User message:\n{user_text}\n\n"
        f"Relevant context (may be in English and may be partial):\n{context}\n\n"
        f"Answer in the user's language (detected: {lang}, role: {role_disp}):"
    )
    logger.info(f"final_prompt(sample 400): {final_prompt[:400]}")

    # 7) LLM
    try:
        answer_markdown = await get_rag_answer(final_prompt)
    except Exception as e:
        logger.exception(f"OpenAI call failed: {e}")
        answer_markdown = (
            "Beklager, der opstod en fejl under genereringen af svaret." if lang == "da"
            else "Sorry, an error occurred while generating the answer."
        )

    answer_plain = md_to_plain(answer_markdown)
    if output_mode == "plain":
        answer_out = answer_plain
    elif output_mode == "tech_brief":
        answer_out = answer_plain  # model instrueres til ren tekst; dette er sikkerhedsnet
    else:
        answer_out = answer_markdown

    # 8) Anti-echo: hvis identisk med nylige svar, vælg safe fallback
    clip = (answer_plain or "").strip()[:400]
    if clip:
        h = hashlib.sha256(clip.encode("utf-8")).hexdigest()
        if h in _LAST_HASHES:
            logger.info("Anti-echo trigger: answer similar to recent output; returning safe generic.")
            safe_generic = {
                "da": (
                    "Konteksten er utilstrækkelig til specifikke tal. Angiv mål (intrusion/extrusion, elastikmønster, IPR-lokation) "
                    "og upload relevante noter/bogmærker, så returnerer jeg en præcis receptblok."
                ),
                "en": (
                    "Context is insufficient for specific parameters. Provide goals (intrusion/extrusion, elastics pattern, IPR location) "
                    "and upload pertinent notes/anchors for a precise prescription block."
                ),
            }
            answer_out = safe_generic.get(lang, safe_generic["en"])
            answer_markdown = answer_out
            answer_plain = answer_out
        _LAST_HASHES.append(h)

    # 9) Sources
    sources = []
    for ch in (qa_hits[:2] + mao_ref_hits[:2] + mao_kw_hits[:1] + history_hits[:1]):
        meta = ch.get("meta", {})
        label = _label_from_meta(meta)
        url = meta.get("url") if isinstance(meta, dict) else None
        src = {"label": label, "url": url}
        if isinstance(meta, dict) and meta.get("refs"):
            src["refs"] = meta.get("refs")
        if isinstance(meta, dict) and (meta.get("anchor_id") or meta.get("anchorId")):
            src["anchor_id"] = meta.get("anchor_id") or meta.get("anchorId")
        sources.append(src)

    return {
        "finalAnswer": answer_out,
        "finalAnswerMarkdown": answer_markdown,
        "finalAnswerPlain": answer_plain,
        "language": lang,
        "role": role_disp,
        "sources": sources,
        "ticketId": ticket_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "message": {"content": answer_out, "language": lang}
    }
