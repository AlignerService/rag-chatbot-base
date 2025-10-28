# app/__init__.py

import os
import json
import logging
import asyncio
import hmac
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from functools import lru_cache
from pathlib import Path

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

from app.db.migrate import run_migrations

# =========================
# [A] Logging & basic config (skal være tidligt)
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("rag-app")

FAISS_INDEX_FILE   = os.getenv("FAISS_INDEX_FILE", "faiss.index")

# -------- Metadata (multi-file support) --------
METADATA_FILE       = os.getenv("METADATA_FILE", "metadata.json")    # fallback (single file)
# Brug KUN METADATA_FILES nedenfor (kommasepareret/whitespace/semikolon)
METADATA_FILES_RAW  = os.getenv("METADATA_FILES", "")

LOCAL_DB_PATH = os.getenv("LOCAL_DB_PATH", "/data/rag.sqlite3")

RAG_BEARER_TOKEN   = os.getenv("RAG_BEARER_TOKEN", "")

OPENAI_MODEL       = os.getenv("OPENAI_CHAT_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
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

# Optional output mode: markdown | plain | tech_brief | mail
OUTPUT_MODE_DEFAULT = os.getenv("OUTPUT_MODE", "markdown").lower()

# ===== Embeddings/rerank config (NYT) =====
EMBED_MODEL   = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
RETR_K        = int(os.getenv("RETR_K", "40"))           # kandidat-mængde før rerank
RERANK_TOPK   = int(os.getenv("RERANK_TOPK", "12"))      # semantisk topK
MMR_LAMBDA    = float(os.getenv("MMR_LAMBDA", "0.4"))    # diversitet 0..1

# Globals
_FAISS_INDEX = None
_METADATA: List[Dict[str, Any]] = []
_SQLITE_OK = False

# Q&A globals
_QA_ITEMS: List[Dict[str, Any]] = []

# ========== BEGIN HOTFIX: SQLite + thread helpers ==========
DB_PATH = LOCAL_DB_PATH  # én sandhed

# Fallback hvis hydrator ikke findes i dette deploy
async def _hydrate_thread_from_zoho(ticket_id: str) -> int:
    try:
        fn = globals().get("zoho_fetch_and_persist_thread")
        if not fn:
            logger.info("Zoho hydrate helper ikke tilgængelig i dette build – springer over.")
            return 0
        n = await fn(ticket_id)
        return n or 0
    except Exception:
        logger.exception("Zoho hydrate failed for %s", ticket_id)
        return 0

async def _table_has(conn, table: str) -> bool:
    async with conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ) as cur:
        return await cur.fetchone() is not None

async def _first_existing_col(conn, table: str, candidates: list[str]) -> str | None:
    async with conn.execute(f"PRAGMA table_info('{table}')") as cur:
        cols = [r[1] async for r in cur]  # r[1] = name
    for c in candidates:
        if c in cols:
            return c
    return None

def _coalesce_cols(cols: list[str]) -> str:
    if not cols:
        return "''"
    parts = [f"TRIM({c})" for c in cols]
    # COALESCE(TRIM(c1), TRIM(c2), ..., '')
    return "COALESCE(" + ", ".join(parts + ["''"]) + ")"

async def _resolve_messages_mapping(conn) -> dict:
    mapping = {}
    table = "messages"
    if not await _table_has(conn, table):
        return {"table": None}

    mapping["table"] = table
    mapping["ticket"]     = await _first_existing_col(conn, table, ["ticketId","ticket_id","tid","ticket"])
    mapping["message_id"] = await _first_existing_col(conn, table, ["message_id","msg_id","id"])
    mapping["subject"]    = await _first_existing_col(conn, table, ["subject","title","emne"])
    mapping["direction"]  = await _first_existing_col(conn, table, ["direction","dir","type"])
    mapping["created"]    = await _first_existing_col(conn, table, ["createdAt","created_at","createdTime","timestamp","date","time","created"])

    # tekstkolonner i prioritet
    candidates = ["body_clean","plainText","plaintext","text","content","body","message","msg"]
    async with conn.execute(f"PRAGMA table_info('{table}')") as cur:
        cols = [r[1] async for r in cur]
    mapping["text_cols"] = [c for c in candidates if c in cols]
    return mapping

def _where_direction_inbound(direction_col: str | None) -> str:
    if not direction_col:
        return "1=1"
    return f"LOWER(TRIM({direction_col})) IN ('in','inbound','received')"

async def _fetch_latest_inbound_from_sqlite(ticket_id: str) -> dict | None:
    if not ticket_id:
        return None
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        mp = await _resolve_messages_mapping(db)
        if not mp.get("table"):
            logger.info("Ingen 'messages' tabel i SQLite – kan ikke hente inbound.")
            return None

        t = mp["table"]
        ticket_col = mp["ticket"]
        direction_col = mp["direction"]
        created_col = mp["created"]
        msg_id_col  = mp["message_id"]
        subj_col    = mp["subject"]
        text_cols   = mp["text_cols"] or []

        if not ticket_col or not text_cols:
            logger.info("Schema mangler ticket/text kolonner – kan ikke hente inbound.")
            return None

        text_expr = _coalesce_cols(text_cols)
        order_by = f"ORDER BY {created_col} DESC" if created_col else "ORDER BY rowid DESC"

        sql = f"""
            SELECT
                {msg_id_col or 'NULL'} AS message_id,
                {subj_col or "''"}   AS subject,
                {text_expr}          AS body_any,
                {created_col or 'NULL'} AS created_at
            FROM {t}
            WHERE {ticket_col} = ?
              AND {_where_direction_inbound(direction_col)}
              AND COALESCE(TRIM({text_cols[0]}), '') <> ''
            {order_by}
            LIMIT 1
        """
        async with db.execute(sql, (ticket_id,)) as cur:
            row = await cur.fetchone()
            if not row:
                return None
            return {
                "message_id": row["message_id"],
                "subject": row["subject"],
                "body_clean": row["body_any"],
                "body": row["body_any"],
                "sender_name": None,
                "created_at": row["created_at"],
            }

async def _fetch_recent_thread_rows(ticket_id: str, limit: int = 12) -> list[dict]:
    if not ticket_id:
        return []
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        mp = await _resolve_messages_mapping(db)
        if not mp.get("table"):
            return []

        t = mp["table"]
        ticket_col = mp["ticket"]
        direction_col = mp["direction"]
        created_col = mp["created"]
        msg_id_col  = mp["message_id"]
        subj_col    = mp["subject"]
        text_cols   = mp["text_cols"] or []

        if not ticket_col or not text_cols:
            return []

        text_expr = _coalesce_cols(text_cols)
        order_by = f"ORDER BY {created_col} DESC" if created_col else "ORDER BY rowid DESC"

        sql = f"""
            SELECT
                {msg_id_col or 'NULL'} AS message_id,
                {subj_col or "''"}   AS subject,
                {text_expr}          AS body_any,
                {direction_col or "''"} AS direction,
                {created_col or 'NULL'} AS created_at
            FROM {t}
            WHERE {ticket_col} = ?
              AND COALESCE(TRIM({text_cols[0]}), '') <> ''
            {order_by}
            LIMIT ?
        """
        out = []
        async with db.execute(sql, (ticket_id, limit)) as cur:
            async for r in cur:
                out.append({
                    "message_id": r["message_id"],
                    "subject": r["subject"],
                    "body_clean": r["body_any"],
                    "body": r["body_any"],
                    "direction": (r["direction"] or "").strip(),
                    "created_at": r["created_at"],
                })
        return out

def _build_context_from_rows(rows: list[dict]) -> list[dict]:
    ctx = []
    for r in rows[:6]:
        ctx.append({
            "id": r.get("message_id"),
            "title": r.get("subject") or "Ticket message",
            "url": None,
            "meta": {"created_at": r.get("created_at"), "direction": r.get("direction")},
            "text": (r.get("body_clean") or r.get("body") or "").strip()
        })
    return ctx

def _inject_greeting(mail_text: str, customer_name: str, lang: str = "da") -> str:
    name = (customer_name or "").strip()
    if not name:
        return mail_text
    lower = mail_text.lower().lstrip()
    if lower.startswith("hej ") or lower.startswith("kære ") or lower.startswith("dear "):
        return mail_text
    greeting = f"Hej {name},\n\n" if lang == "da" else f"Hi {name},\n\n"
    return greeting + mail_text
# ========== END HOTFIX ==========


# =========================
# [B] FastAPI & CORS (app kan godt komme før helpers; bare ikke router-import)
# =========================
app = FastAPI()

WIX_ORIGIN = os.getenv("WIX_ORIGIN", "https://www.alignerservice.com").strip()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[WIX_ORIGIN],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Content-Type", "X-Chat-Token", "Authorization"],
)


# =========================
# [C] Security
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
# [D] Startup
# =========================
@app.get("/healthz/sqlite")
async def healthz_sqlite():
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("PRAGMA journal_mode=WAL;")
            async with db.execute("SELECT 1") as cur:
                await cur.fetchone()
        return {"ok": True, "db_path": DB_PATH}
    except Exception as e:
        return {"ok": False, "db_path": DB_PATH, "error": str(e)}

@app.post("/debug/sqlite-write-read")
async def debug_sqlite_write_read():
    now = datetime.utcnow().isoformat()+"Z"
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("""CREATE TABLE IF NOT EXISTS messages_debug(
                id INTEGER PRIMARY KEY,
                created_at TEXT,
                note TEXT
            )""")
            await db.execute("INSERT INTO messages_debug(created_at, note) VALUES(?, ?)", (now, "ping"))
            await db.commit()
            async with db.execute("SELECT created_at, note FROM messages_debug ORDER BY id DESC LIMIT 5") as cur:
                rows = await cur.fetchall()
        return {"ok": True, "wrote": now, "recent": [tuple(r) for r in rows]}
    except Exception as e:
        return {"ok": False, "error": str(e), "db_path": DB_PATH}

@app.get("/debug/sqlite-cols")
async def debug_sqlite_cols():
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            cols = []
            async with db.execute("PRAGMA table_info('chat_sessions')") as cur:
                async for r in cur:
                    # r: (cid, name, type, notnull, dflt_value, pk)
                    cols.append({"name": r[1], "type": r[2], "pk": bool(r[5])})
        return {"ok": True, "table": "chat_sessions", "columns": cols}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.on_event("startup")
async def on_startup():
    if RAG_BEARER_TOKEN:
        logger.info("RAG_BEARER_TOKEN is set (value hidden)")
    else:
        logger.error("RAG_BEARER_TOKEN is missing!")

    # ensure dir for local db
    try:
        base = os.path.dirname(LOCAL_DB_PATH)
        if base:
            os.makedirs(base, exist_ok=True)
    except Exception:
        pass

    try:
        # Seed kun første gang / hvis filen mangler
        db_path = Path(LOCAL_DB_PATH)
        if not db_path.exists():
            logger.info("Local DB missing; seeding from Dropbox...")
            await download_db()
        else:
            logger.info(f"Local DB present at {db_path}; skipping Dropbox seed")

        # --- Migrations: soft-fail i produktion ---
        try:
            logger.info("Running DB migrations...")
            if asyncio.iscoroutinefunction(run_migrations):
                await run_migrations()
            else:
                await asyncio.to_thread(run_migrations)
            logger.info("DB migrations complete.")
        except Exception:
            logger.exception("Migration failed; continuing without schema changes")

        # --- ensure UI log table exists ---
        try:
            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS chat_frontend_log(
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        created_at TEXT NOT NULL,
                        kind TEXT,
                        session_id TEXT,
                        email TEXT,
                        text TEXT
                    )
                """)
                await db.commit()
            logger.info("chat_frontend_log table ready.")
        except Exception as e:
            logger.exception("Failed preparing chat_frontend_log: %s", e)      
        
        await init_db()

        # --- load Q&A JSON (unchanged) ---
        global _QA_ITEMS
        _QA_ITEMS = _qa_load_items()
        logger.info("RAG startup complete")
    except Exception as e:
        logger.exception(f"Startup failed: {e}")


# =========================
# [E] Language detection
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

    # sv (svensk) – simpel men effektiv heuristik
    if sum(1 for w in [" hej ", " vänligen", " önskar", " mellanrum", " krona", " regio", "tand ", "godkännande", "setup"] if w in lowered) >= 2:
        return "sv"

    return "en"


# =========================
# [F] Intent detection
# =========================
def detect_intent(text: str, lang: str) -> str:
    t = (text or "").lower()
    if lang == "da":
        status_keys = ["status", "opdatering", "hvornår", "forventet", "eta", "ticket", "case", "sag", "case id", "reference"]
        admin_keys  = ["faktura", "betaling", "konto", "login", "adgang", "levering", "fragt", "retur", "reklamation", "pris", "prisliste"]
    elif lang == "de":
        status_keys = ["status", "update", "wann", "eta", "ticket", "fall", "case", "referenz"]
        admin_keys  = ["rechnung", "zahlung", "konto", "login", "zugang", "lieferung", "versand", "retoure", "beschwerde", "preis"]
    else:
        status_keys = ["status", "update", "when", "eta", "ticket", "case", "reference", "order"]
        admin_keys  = ["invoice", "payment", "account", "login", "access", "delivery", "shipping", "return", "complaint", "price", "pricelist"]

    if any(k in t for k in status_keys):
        return "status_request"
    if any(k in t for k in admin_keys):
        return "admin"
    return "clinical_support"


# =========================
# [G] Customer-facing prompts
# =========================
def make_customer_system_prompt(lang: str) -> str:
    if lang == "da":
        return (
            "Du skriver til en professionel kunde (tandlæge/ortodontist). Svar kort, klart og venligt. "
            "For status/admin: ingen kliniske instruktioner. Bekræft anmodning, angiv næste skridt, og bed kun om nødvendige oplysninger.\n\n"
            "SIKKERHED\n• Del ikke adgangskoder eller interne systemoplysninger. • Ingen patientnavne. • Brug kun case-id som kunden selv nævner.\n"
        )
    return (
        "You are writing to a professional customer (dentist/orthodontist). Keep it brief and clear. "
        "For status/admin: no clinical instructions. Acknowledge, give next steps, ask only for necessary fields.\n\n"
        "SAFETY\n• No passwords or internal system details. • No patient names. • Use only the case ID the customer provided.\n"
    )


# =========================
# [H] Brand & case-id detection
# =========================
_BRAND_PATTERNS = [
    ("Invisalign",   re.compile(r"\b(\d{8})\b")),          # e.g. 26034752
    ("Spark",        re.compile(r"\b(\d{7})\b")),          # e.g. 2928247
    ("Angel Aligner",re.compile(r"\b([A-Z0-9]{6})\b")),     # e.g. 97KP8K
    ("ClearCorrect", re.compile(r"\b(\d{7})\b")),          # e.g. 2027367
    ("SureSmile",    re.compile(r"\b([A-Z0-9]{4})\b")),     # e.g. J8U3
    ("TrioClear",    re.compile(r"\b(\d{5})\b")),          # e.g. 45447
    # Clarity: mangler sikre eksempler -> ikke aktiveret mønster endnu
]

_BRAND_ALIASES = {
    "invisalign": "Invisalign",
    "spark": "Spark",
    "angel": "Angel Aligner",
    "angel aligner": "Angel Aligner",
    "clearcorrect": "ClearCorrect",
    "sure smile": "SureSmile",
    "suresmile": "SureSmile",
    "trioclear": "TrioClear",
    "clarity": "Clarity",
}

def _normalize_brand(s: str) -> Optional[str]:
    if not s:
        return None
    t = s.strip().lower()
    return _BRAND_ALIASES.get(t) or None

def extract_brand_and_case(subject: str, body: str) -> Dict[str, Optional[str]]:
    txt = f"{subject or ''}\n{body or ''}"
    low = txt.lower()

    brand_hint = None
    for alias, canon in _BRAND_ALIASES.items():
        if alias in low:
            brand_hint = canon
            break

    found_brand, found_case = None, None

    if brand_hint:
        for name, pat in [(n,p) for (n,p, *_) in _BRAND_PATTERNS]:
            if name == brand_hint:
                m = pat.search(txt.upper() if name in ("Angel Aligner", "SureSmile") else txt)
                if m:
                    found_brand, found_case = name, m.group(1)
                break

    if not found_case:
        hits = []
        for name, pat in [(n,p) for (n,p, *_) in _BRAND_PATTERNS]:
            m = pat.search(txt.upper() if name in ("Angel Aligner", "SureSmile") else txt)
            if m:
                hits.append((name, m.group(1)))
        if len(hits) == 1:
            found_brand, found_case = hits[0]

    return {"brand": found_brand, "caseId": found_case}


# =========================
# [I] Role detection
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
    }
    lang_map = lex.get(lang, lex["en"])
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
    }
    return labels.get(lang, labels["en"]).get(role, role)


# =========================
# [J] Persona prompt (role-aware)
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
            "• Ingen direkte patienthenvendelse. Opret aldrig interne detaljer, der ikke er i konteksten.\n\n"
            "Når output_mode er 'mail': skriv som komplet e-mail i ren tekst og medtag ikke kilder eller sprogfelter i brødteksten."
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
        "• Do not address patients. Never invent internal details not in context.\n\n"
        "When output_mode is 'mail': write a complete email in plain text and do not include sources or language fields in the body."
    )


# =========================
# [K] Dropbox download
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
# [L] Metadata loader
# =========================
def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def _json_to_list_any(txt: str) -> List[Dict[str, Any]]:
    import json
    def only_objs(seq):
        return [x for x in seq if isinstance(x, dict)]

    try:
        obj = json.loads(txt)
        if isinstance(obj, list):
            return only_objs(obj)
        if isinstance(obj, dict):
            for k in ("chunks","items","data","documents","rows"):
                if isinstance(obj.get(k), list):
                    return only_objs(obj[k])
            flat = []
            for v in obj.values():
                if isinstance(v, list):
                    flat.extend(v)
            if flat:
                return only_objs(flat)
    except Exception:
        pass

    out = []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            if isinstance(rec, dict):
                out.append(rec)
        except Exception:
            continue
    if out:
        return out

    return []

def _load_metadata_files() -> List[Dict[str, Any]]:
    paths_env = METADATA_FILES_RAW.strip()
    single = METADATA_FILE.strip()
    paths: List[str] = []

    if paths_env:
        for part in re.split(r"[,\s;]+", paths_env):
            if part:
                paths.append(part)
    elif single:
        paths = [single]

    if not paths:
        logger.info("No metadata files configured (METADATA_FILES or METADATA_FILE).")
        return []

    loaded: List[Dict[str, Any]] = []
    for p in paths:
        try:
            if not os.path.exists(p):
                logger.warning(f"Metadata file not found: {p}")
                continue
            txt = _read_text(p)
            lst = _json_to_list_any(txt)
            if not lst:
                logger.warning(f"Metadata file {p} did not contain a list; skipping.")
                continue
            loaded.extend(lst)
            logger.info(f"Loaded {len(lst)} metadata items from {p}")
        except Exception as e:
            logger.exception(f"Failed loading metadata '{p}': {e}")

    logger.info(f"Total metadata items loaded: {len(loaded)}")
    return loaded


# =========================
# [M] Init FAISS / metadata / SQLite
# =========================
async def init_db():
    global _FAISS_INDEX, _METADATA, _SQLITE_OK

    # FAISS
    if faiss is not None and os.path.exists(FAISS_INDEX_FILE):
        try:
            _FAISS_INDEX = faiss.read_index(FAISS_INDEX_FILE)
            logger.info(f"Loaded FAISS index from {FAISS_INDEX_FILE}")
        except Exception as e:
            logger.exception(f"Failed to load FAISS index: {e}")
            _FAISS_INDEX = None
    else:
        logger.info("FAISS not available or index missing; continuing without FAISS.")

    # METADATA
    try:
        _METADATA = _load_metadata_files()
        if not _METADATA:
            logger.info("No metadata loaded (check METADATA_FILES or METADATA_FILE).")
    except Exception as e:
        logger.exception(f"Failed to load metadata: {e}")
        _METADATA = []

    # SQLite ping
    try:
        if os.path.exists(DB_PATH):
            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute("SELECT 1")
            _SQLITE_OK = True
            logger.info(f"SQLite available at {DB_PATH}")
        else:
            _SQLITE_OK = False
            logger.info(f"SQLite path not found: {DB_PATH} (continuing)")
    except Exception as e:
        _SQLITE_OK = False
        logger.exception(f"SQLite check failed: {e}")

    # === NYT: Indlæs historik som RAG-chunks fra SQLite ===
    try:
        if _SQLITE_OK:
            hist_chunks = await sqlite_load_threads_as_chunks(limit=3000)
            if hist_chunks:
                _METADATA.extend(hist_chunks)
                logger.info(f"Loaded {len(hist_chunks)} history chunks into metadata")
    except Exception as e:
        logger.info(f"history load skipped: {e}")


# =========================
# [N] Text helpers (used by SQLite functions)
# =========================
def _strip_html(text: str) -> str:
    if not text:
        return ""
    import re
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =========================
# [O] SQLite helpers
# =========================
async def sqlite_latest_thread_plaintext(ticket_id: str, contact_id: Optional[str] = None) -> str:
    if not _SQLITE_OK:
        logger.warning("SQLite not available")
        return ""
    tid = (ticket_id or "").strip()
    cid = (contact_id or "").strip() if contact_id else None
    if not tid:
        return ""
    text_cols   = ["plainText","plaintext","text","content","body","message","msg"]
    ticket_cols = ["ticketId","ticket_id","tid","ticket"]
    contact_cols= ["contactId","contact_id","cid","contact"]
    time_cols   = ["createdTime","createdAt","created_at","timestamp","date","time"]
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT name FROM sqlite_master WHERE type='table'") as c:
                tables=[r["name"] async for r in c]
            best_txt,best_when="",""
            for t in tables:
                try:
                    async with db.execute(f"PRAGMA table_info('{t}')") as c:
                        cols=[r["name"] async for r in c]
                    tcol=next((x for x in text_cols if x in cols),None)
                    kcol=next((x for x in ticket_cols if x in cols),None)
                    ccol=next((x for x in contact_cols if x in cols),None)
                    dcol=next((x for x in time_cols if x in cols),None)
                    if not (tcol and kcol): continue
                    where=[f"{kcol}=?"]; p=[tid]
                    if cid and ccol: where.append(f"{ccol}=?"); p.append(cid)
                    where.append(f"COALESCE(TRIM({tcol}), '')<>''")
                    order=f"ORDER BY {dcol} DESC" if dcol else "ORDER BY rowid DESC"
                    sql=f"SELECT {tcol} AS txt,{dcol if dcol else 'NULL'} AS dt FROM '{t}' WHERE {' AND '.join(where)} {order} LIMIT 1"
                    async with db.execute(sql,p) as c2:
                        r=await c2.fetchone()
                    if not r or not r['txt']: continue
                    if not best_when or (r['dt'] and str(r['dt'])>str(best_when)):
                        best_txt,best_when=r['txt'],r['dt']
                except Exception: continue
            return best_txt or ""
    except Exception as e:
        logger.exception(f"sqlite_latest_thread_plaintext failed: {e}")
        return ""

async def sqlite_load_threads_as_chunks(limit:int=2000)->List[Dict[str,Any]]:
    if not _SQLITE_OK: return []
    out=[]
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory=aiosqlite.Row
            sql="""SELECT m.plainText AS txt,m.ticketId,t.contactId,t.subject,c.email
                   FROM messages m
                   LEFT JOIN tickets t ON t.ticketId=m.ticketId
                   LEFT JOIN contacts c ON c.contactId=t.contactId
                   WHERE COALESCE(TRIM(m.plainText),'')<>''
                   ORDER BY m.createdAt DESC LIMIT ?"""
            async with db.execute(sql,(limit,)) as cur:
                async for r in cur:
                    txt=(r["txt"] or "").strip()
                    if len(txt)<40: continue
                    out.append({"text":txt,"meta":{"ticketId":r["ticketId"],"subject":r["subject"],"contactId":r["contactId"],"contact_email":r["email"]}})
    except Exception as e: logger.info(f"sqlite_load_threads_as_chunks: {e}")
    return out

async def sqlite_style_snippets(to_email: str, n: int = 4, min_len: int = 120) -> List[str]:
    if not _SQLITE_OK or not to_email:
        return []

    want = (to_email or "").strip().lower()
    out: List[str] = []

    try:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row

            sql_by_email = """
                SELECT m.plainText AS txt
                FROM contacts c
                JOIN tickets t ON t.contactId = c.contactId
                JOIN messages m ON m.ticketId = t.ticketId
                WHERE LOWER(TRIM(c.email)) = ?
                  AND COALESCE(TRIM(m.plainText),'') <> ''
                  AND LOWER(TRIM(m.direction)) IN ('out','outbound','sent')
                ORDER BY
                    CASE WHEN m.createdAt GLOB '____-__-__*' THEN m.createdAt ELSE NULL END DESC,
                    m.rowid DESC
                LIMIT ?
            """
            async with db.execute(sql_by_email, (want, max(n*5, n))) as cur:
                async for r in cur:
                    t = _strip_html((r["txt"] or "").strip())
                    if len(t) >= min_len:
                        out.append(t[:600])
                        if len(out) >= n:
                            break

            if len(out) < n:
                sql_any_out = """
                    SELECT m.plainText AS txt
                    FROM messages m
                    WHERE COALESCE(TRIM(m.plainText),'') <> ''
                      AND LOWER(TRIM(m.direction)) IN ('out','outbound','sent')
                    ORDER BY
                        CASE WHEN m.createdAt GLOB '____-__-__*' THEN m.createdAt ELSE NULL END DESC,
                        m.rowid DESC
                    LIMIT ?
                """
                async with db.execute(sql_any_out, (max(n*5, n),)) as cur:
                    async for r in cur:
                        t = _strip_html((r["txt"] or "").strip())
                        if len(t) >= min_len:
                            out.append(t[:600])
                            if len(out) >= n:
                                break

        uniq: List[str] = []
        seen: set = set()
        for s in out:
            k = s[:120].lower()
            if k not in seen:
                uniq.append(s)
                seen.add(k)
            if len(uniq) >= n:
                break
        return uniq

    except Exception as e:
        logger.info(f"sqlite_style_snippets (messages-based): {e}")
        return []


# =========================
# [P] Text helpers
# =========================
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

def _label_from_meta(m):
    if isinstance(m, dict):
        return m.get("title") or m.get("source") or m.get("path") or m.get("url") or m.get("id") or "metadata"
    return "metadata"

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

def _deidentify(txt: str) -> str:
    if not txt:
        return txt
    s = txt
    s = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[email]", s)
    s = re.sub(r"\+?\d[\d\-\s]{6,}\d", "[phone]", s)
    s = re.sub(r"\b\d{6}\-?\d{4}\b", "[id]", s)
    s = re.sub(r"\b([A-ZÆØÅ][a-zæøå]{2,})(\s+[A-ZÆØÅ][a-zæøå]{2,}){0,2}\b", "[name]", s)
    return s


# =========================
# [Q] Klinisk faktekstraktor og boosters
# =========================
def extract_facts(text: str) -> Dict[str, Any]:
    t = (text or "").lower()
    facts = {
        "angle_class": "unknown",
        "crossbite": any(x in t for x in ["crossbite","krydsbid"]),
        "open_bite": any(x in t for x in ["open bite","åbent bid","åben bid"]),
        "deep_bite": any(x in t for x in ["deep bite","dybt bid"]),
        "bolton": "bolton" in t,
        "crowding_mm": None,
        "spacing_mm": None,
        "overjet_mm": None,
        "overbite_mm": None,
        "habits": any(x in t for x in ["tongue thrust","tungepres","mouth breathing","mundånding"]),
    }
    if "class ii" in t or "klasse ii" in t: facts["angle_class"] = "II"
    elif "class iii" in t or "klasse iii" in t: facts["angle_class"] = "III"
    elif "class i" in t or "klasse i" in t: facts["angle_class"] = "I"
    for k, pat in [("crowding_mm", r"crowding[^0-9]{0,8}(\d+(?:\.\d+)?)\s*mm"),
                   ("spacing_mm",  r"spacing[^0-9]{0,8}(\d+(?:\.\d+)?)\s*mm"),
                   ("overjet_mm",  r"overjet[^0-9]{0,8}(\d+(?:\.\d+)?)\s*mm"),
                   ("overbite_mm", r"overbite[^0-9]{0,8}(\d+(?:\.\d+)?)\s*mm")]:
        m = re.search(pat, t)
        if m: facts[k] = float(m.group(1))
    return facts

def build_query_boosters(facts: Dict[str, Any]) -> List[str]:
    out = []
    ac = facts.get("angle_class")
    if ac and ac!="unknown": out += [f"angle class {ac}", f"klasse {ac}"]
    if facts.get("open_bite"): out += ["anterior open bite","åbent bid","vertical discrepancy","intrusion"]
    if facts.get("deep_bite"): out += ["deep bite","dybt bid","extrusion","incisal display"]
    if facts.get("crossbite"): out += ["crossbite","krydsbid","transverse"]
    if facts.get("bolton"): out += ["bolton analysis","bolton ratio"]
    return list(dict.fromkeys(out))

def _label_join(meta: Dict[str, Any], text: str) -> str:
    return (_label_from_meta(meta) + "|" + (_strip_html(text)[:120])).lower()


# =========================
# [R] Embeddings + rerank + MMR
# =========================
@lru_cache(maxsize=4096)
def _embed_cached(text: str) -> List[float]:
    from openai import OpenAI  # type: ignore
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", OPENAI_API_KEY))
    emb = client.embeddings.create(model=EMBED_MODEL, input=text)
    return emb.data[0].embedding  # type: ignore

def _cos(u: List[float], v: List[float]) -> float:
    import math
    if not u or not v or len(u)!=len(v): return 0.0
    su = math.sqrt(sum(x*x for x in u)); sv = math.sqrt(sum(x*x for x in v))
    if su==0 or sv==0: return 0.0
    return sum(a*b for a,b in zip(u,v)) / (su*sv)

def semantic_rerank(query: str, candidates: List[Dict[str, Any]], topk: int) -> List[Dict[str, Any]]:
    if not candidates: return []
    try:
        q_emb = _embed_cached(query)
        scored = []
        for c in candidates:
            txt = _strip_html(_extract_text_from_meta(c))
            emb = _embed_cached(txt[:2000])
            scored.append(( _cos(q_emb, emb), c))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:topk]]
    except Exception as e:
        logger.info(f"semantic_rerank fallback: {e}")
        return candidates[:topk]

def mmr_select(query: str, items: List[Dict[str, Any]], lam: float, m: int) -> List[Dict[str, Any]]:
    if not items: return []
    try:
        q_emb = _embed_cached(query)
        embs = []
        for it in items:
            txt = _strip_html(_extract_text_from_meta(it))
            embs.append(_embed_cached(txt[:2000]))
        selected, used = [], set()
        sims_q = [ _cos(q_emb, e) for e in embs ]
        while len(selected) < min(m, len(items)):
            best_i, best_score = -1, -1.0
            for i in range(len(items)):
                if i in used: continue
                div_penalty = 0.0
                if selected:
                    div_penalty = max(_cos(embs[i], embs[j]) for j in selected)
                score = lam * sims_q[i] - (1-lam) * div_penalty
                if score > best_score:
                    best_score, best_i = score, i
            used.add(best_i)
            selected.append(best_i)
        return [items[i] for i in selected]
    except Exception as e:
        logger.info(f"mmr_select fallback: {e}")
        return items[:m]


# =========================
# [S] Style hints & policy helpers
# =========================
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
        if mode == "mail":
            return (
                "MAILFORMAT (dansk): Svar som en færdig e-mail i REN TEKST (ingen markdown, ingen emojis). "
                "Hold det kort, klinisk og kollegialt. Ingen 'Sources', 'Language' eller debug-felter i brødteksten.\n\n"
                "Brug denne skabelon PRÆCIST (inkl. sektionstitler):\n"
                "Emne: <kort, præcis emnelinje>\n\n"
                "Hej,\n\n"
                "Kort konklusion: <1–2 linjer med hovedbudskab>\n\n"
                "Plan:\n"
                "1) <punkt>\n"
                "2) <punkt>\n"
                "3) <punkt>\n"
                "4) <punkt (valgfrit)>\n\n"
                "Hvad vi har brug for:\n"
                "- <liste over materiale/fotos/IO-scan osv.>\n\n"
                "Næste skridt:\n"
                "- <konkret CTA: send materialet / foreslå mødetid / vi fremsender IPR-kort osv.>\n\n"
                "Bedste hilsner\n"
                "[Din signatur]\n\n"
                "VIGTIGT: Brug kun information fra 'Relevant context'. Vær forsigtig med kategoriske ordinationer; "
                "brug 'foreslår/kan overvejes' og angiv, at endelig plan afhænger af kliniske fund og compliance."
            )
        return ""  # markdown
    else:
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
        if mode == "mail":
            return (
                "MAIL FORMAT (English): Write a complete email in PLAIN TEXT (no markdown, no emojis). "
                "Keep it concise, clinical, and collegial. Do NOT include 'Sources', 'Language', or debug fields in the body.\n\n"
                "Use this exact structure:\n"
                "Subject: <short, precise subject>\n\n"
                "Hi,\n\n"
                "Brief conclusion: <1–2 lines with the key takeaway>\n\n"
                "Plan:\n"
                "1) <item>\n"
                "2) <item>\n"
                "3) <item>\n"
                "4) <optional item>\n\n"
                "What we need:\n"
                "- <list: photos/IO-scan/etc>\n\n"
                "Next steps:\n"
                "- <clear CTA: send assets / book time / we will send IPR map>\n\n"
                "Best regards\n"
                "[Your signature]\n\n"
                "IMPORTANT: Use only the 'Relevant context'. Avoid categorical prescriptions; "
                "prefer 'suggest/consider' and note that the final plan depends on clinical findings and compliance."
            )
        return ""  # markdown


# =========================
# [T] Q&A JSON search
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
_QA_WEIGHTS.update({
    "angle": 1.5, "klasse": 1.5, "i": 0.6, "ii": 1.0, "iii": 1.0,
    "vertical": 1.6, "mpa": 1.6, "bolton": 2.2, "overjet": 1.2, "overbite": 1.2,
    "open-bite": 2.0, "aob": 2.0
})

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
# [U] Retrieval helpers + MAO helpers
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
# [V] LLM translation & calls
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
            temperature=0.0,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e_modern:
        logger.exception(f"Short OpenAI call failed: {e_modern}")
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
            temperature=0.2,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e_modern:
        logger.exception(f"OpenAI call failed: {e_modern}")
        return "I could not generate a response due to an internal error."


# =========================
# [W] Policy helpers
# =========================
def stop_orders_block(lang: str) -> str:
    if lang == "da":
        return ("POLICY: Ingen adgangskoder/tokens/private links. Ingen patientnavne. "
                "Ingen deling af case-id på tværs af kunder. Ingen kliniske anvisninger der ændrer biologi "
                "(TADs, >1–2 mm distalisation, ekspansion, specifik IPR) uden tilstrækkelige data (Bolton, crowding/spacing mm, OJ/OB).")
    return ("POLICY: No passwords/tokens/private links. No patient names. No cross-customer case IDs. "
            "No invasive clinical instructions without sufficient data.")

def customer_mail_guidance(lang: str, intent: str) -> str:
    if lang == "da":
        if intent == "status_request":
            return "MAIL: 3 blokke — kvittering, kort status, næste skridt. Ingen kliniske instruktioner."
        if intent == "admin":
            return "MAIL: 1–2 linjer forklaring + 1–3 konkrete trin eller link. Ingen kliniske instruktioner."
        return ""
    else:
        if intent == "status_request":
            return "MAIL: 3 blocks — receipt, brief status, next steps. No clinical instructions."
        if intent == "admin":
            return "MAIL: 1–2 lines + 1–3 concrete steps or link. No clinical instructions."
        return ""

def fallback_mail(lang: str, intent: str) -> str:
    if lang != "da":
        lang = "da"
    if intent == "status_request":
        return ("Tak for din henvendelse. For at tjekke status skal vi bruge case-ID og hvilket alignerbrand der er tale om "
                "(fx Invisalign, Spark eller ClearCorrect). Send det gerne, så følger vi op hurtigst muligt.\n\n"
                "Venlig hilsen\nTandlæge Helle Hatt")
    if intent == "admin":
        return ("Tak for beskeden. Angiv venligst hvad der skal ændres (fx levering, faktura, adgang) og evt. reference, "
                "så vender vi hurtigt tilbage.\n\nVenlig hilsen\nTandlæge Helle Hatt")
    return ("Tak for din besked. For at give et konkret klinisk svar må du gerne sende relevante fotos/IOS-scan samt kort beskrivelse "
            "af udfordringen (fx tracking/rotation/åbent bid). Vi vender tilbage inden for 24 timer på hverdage.\n\n"
            "Venlig hilsen\nTandlæge Helle Hatt")


# =========================
# [X] Endpoints
# =========================
@app.get("/")
async def health():
    return {"status": "ok", "qa_loaded": len(_QA_ITEMS), "metadata_loaded": len(_METADATA)}

@app.post("/debug-token")
async def debug_token(request: Request):
    return JSONResponse({"received_authorization": request.headers.get("authorization")})

@app.post("/debug/ui-log")
async def ui_log(request: Request):
    """
    Lille, sikker log-endpoint til Wix-UI. Gemmer kun ufarlige felter.
    Body forventes som JSON: { "kind": "...", "sessionId": "...", "email": "...", "text": "..." }
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    when = datetime.utcnow().isoformat() + "Z"
    kind = str((body.get("kind") or "")).strip()[:40]
    sid  = str((body.get("sessionId") or "")).strip()[:80]
    mail = str((body.get("email") or "")).strip()[:120]
    text = str((body.get("text") or "")).strip()[:4000]

    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "INSERT INTO chat_frontend_log(created_at, kind, session_id, email, text) VALUES (?, ?, ?, ?, ?)",
                (when, kind, sid, mail, text)
            )
            await db.commit()
        return {"ok": True, "stored": when}
    except Exception as e:
        logger.exception("ui_log insert failed: %s", e)
        raise HTTPException(status_code=500, detail="log write failed")

@app.get("/debug/ui-log")
async def ui_log_tail(n: int = 20):
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT created_at, kind, session_id, email, substr(text,1,200) AS text "
                "FROM chat_frontend_log ORDER BY id DESC LIMIT ?",
                (max(1, min(n, 200)),)
            ) as cur:
                rows = [dict(r) for r in await cur.fetchall()]
        return {"ok": True, "rows": rows}
    except Exception as e:
        logger.exception("ui_log tail failed: %s", e)
        raise HTTPException(status_code=500, detail="log read failed")

@app.post("/api/answer", dependencies=[Depends(require_rag_token)])
async def api_answer(request: Request):
    try:
        body = await request.json()
    except Exception as e:
        logger.exception(f"Invalid JSON body: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    logger.info(f"Incoming body keys: {list(body.keys()) if isinstance(body, dict) else type(body)}")

    output_mode = OUTPUT_MODE_DEFAULT
    if isinstance(body, dict) and "output_mode" in body:
        output_mode = str(body.get("output_mode") or OUTPUT_MODE_DEFAULT).lower()

    from_email = (body.get("fromEmail") or "").strip().lower() if isinstance(body, dict) else ""
    subject = (body.get("subject") or "").strip() if isinstance(body, dict) else ""
    customer_name_hint = (body.get("contactName") or body.get("customerName") or "").strip()

    user_text = ""
    ticket_id: Optional[str] = None
    contact_id: Optional[str] = None
    lang: Optional[str] = None

    if isinstance(body, dict):
        if "ticketId" in body and "question" in body:
            ticket_id = str(body.get("ticketId") or "").strip()
            contact_id = str(body.get("contactId") or "").strip() if body.get("contactId") else None

            txt = await sqlite_latest_thread_plaintext(ticket_id, contact_id)
            if txt:
                user_text = txt
                logger.info("Zoho ID-only: pulled latest thread from SQLite.")
                lang = detect_language(user_text)
            else:
                logger.info("SQLite tom for ticket %s – prøver Zoho hydrate...", ticket_id)
                n = await _hydrate_thread_from_zoho(ticket_id)
                logger.info("Zoho hydrate indlæste %s beskeder", n)
                latest_after = await _fetch_latest_inbound_from_sqlite(ticket_id)
                if latest_after:
                    user_text = (latest_after.get("body_clean") or latest_after.get("body") or "").strip()
                    logger.info("Pulled inbound efter hydrate – bruger kundens mailtekst.")
                    if not subject:
                        subject = (latest_after.get("subject") or subject or "").strip()
                    lang = detect_language(user_text)
                else:
                    user_text = str(body.get("question") or "").strip()
                    logger.warning("Ingen inbound efter hydrate – falder tilbage til 'question' feltet.")

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

    is_boiler = (
        bool(re.search(r"please provide the best ai[- ]generated reply", (user_text or ""), re.I))
        or len((user_text or "").strip()) < 15
    )

    if is_boiler:
        customer_name = ""
        for k in ("customerName","contactName","name"):
            if isinstance(body, dict) and isinstance(body.get(k), str) and body.get(k).strip():
                customer_name = body[k].strip()
                break

        base = (
            "Subject: Tak for din besked\n\n"
            "Hej,\n\n"
            "Tak for din besked – vi har registreret den og går i gang. "
            "Du hører fra os, så snart der er et oplæg klar til godkendelse, "
            "eller hvis vi mangler noget for at komme videre.\n\n"
            "Venlig hilsen\n"
            "AlignerService Team"
        )
        answer = _inject_greeting(base, customer_name, "da") if customer_name else base

        return {
            "finalAnswer": answer,
            "finalAnswerMarkdown": answer,
            "finalAnswerPlain": md_to_plain(answer),
            "language": "da",
            "role": "kliniker",
            "sources": [],
            "ticketId": ticket_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "message": {"content": answer, "language": "da"}
        }

    if not lang:
        lang = detect_language(user_text)

    role = detect_role(user_text, lang)
    role_disp = role_label(lang, role)
    intent = detect_intent(user_text, lang)

    if intent in ("status_request", "admin") and not (isinstance(body, dict) and "output_mode" in body):
        output_mode = "mail"

    system_prompt = make_customer_system_prompt(lang) if intent in ("status_request", "admin") else make_system_prompt(lang, role)

    brand_case = extract_brand_and_case(subject, user_text) if os.getenv("BRAND_ENABLE","1") == "1" else {"brand": None, "caseId": None}

    retrieval_query = await translate_to_english_if_needed(user_text, lang)
    if retrieval_query != user_text:
        logger.info("Query translated to English for retrieval.")
        logger.info(f"retrieval_query(sample 200): {retrieval_query[:200]}")

    facts = extract_facts(user_text)
    boosters = build_query_boosters(facts)
    boosted_query = (retrieval_query + " " + " ".join(boosters)).strip()

    qa_items = search_qa_json(boosted_query, k=RETR_K)
    qa_hits_raw = [_qa_to_chunk(x) for x in qa_items]

    qa_refs = [aid for it in qa_items for aid in (it.get("refs") or []) if isinstance(aid, str) and aid.startswith("MAO:")]
    mao_ref_hits_raw = find_mao_by_anchors(qa_refs, k_per_anchor=2) if qa_refs else []

    mao_kw_hits_raw = []
    try:
        mao_kw_hits_raw = await get_mao_top_chunks(boosted_query, k=max(4, RETR_K//3))
    except Exception as e:
        logger.info(f"MAO keyword retrieval failed: {e}")

    cands = qa_hits_raw + mao_ref_hits_raw + mao_kw_hits_raw
    seen = set(); uniq=[]
    for c in cands:
        txt = _extract_text_from_meta(c)
        meta = c.get("meta", {})
        key = _label_join(meta, txt)
        if key not in seen and txt:
            uniq.append(c); seen.add(key)

    sem_top = semantic_rerank(boosted_query, uniq, topk=RERANK_TOPK)
    diversified = mmr_select(boosted_query, sem_top, lam=MMR_LAMBDA, m=min(8, len(sem_top)))

    history_hits = []
    if ticket_id:
        hist = await sqlite_latest_thread_plaintext(ticket_id, contact_id)
        if hist:
            history_hits = [{"text": hist[:1500], "meta": {"source": "History", "title": "Recent thread"}}]

    def text_of(ch):
        return ch.get("text","") if isinstance(ch,dict) else str(ch)

    def roughly_matches(q, hist_txt):
        ql = q.lower()
        ht = (hist_txt or "").lower()
        toks = [t for t in re.split(r"[^a-z0-9æøåöüß]+", ql) if len(t)>3]
        hit = sum(1 for t in set(toks) if t in ht)
        return hit >= 2

    history_primary = []
    for h in history_hits:
        if roughly_matches(retrieval_query, text_of(h)):
            history_primary.append(h)

    combined_chunks = history_primary + diversified if history_primary else diversified + history_hits

    if not combined_chunks:
        fb = fallback_mail(lang, intent) if intent in ("status_request","admin","clinical_support") else ""
        answer = fb or (
            "Grundprotokol når kontekst mangler:\n"
            "1) Screening/diagnose → 2) Plan (staging, IPR, attachments) → 3) Start → 4) Kontroller/tracking → 5) Refinement → 6) Retention.\n"
            "Næste skridt: indlæs flere kildedata i RAG for mere målrettet svar."
        )
        if lang == "da":
            answer = _inject_greeting(answer, customer_name_hint, "da")
        return {
            "finalAnswer": answer,
            "finalAnswerMarkdown": answer,
            "finalAnswerPlain": md_to_plain(answer),
            "language": lang,
            "role": role_disp,
            "sources": [],
            "ticketId": ticket_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "message": {"content": answer, "language": lang}
        }

    def _clip(txt: str, max_len: int = 1200) -> str:
        return (txt[:max_len] + "…") if len(txt) > max_len else txt

    context = "\n\n".join(_clip(_deidentify(text_of(ch))) for ch in combined_chunks)[:9000]

    style = style_hint(output_mode, lang)
    policy_block = stop_orders_block(lang)

    use_style = bool(from_email and "@" in from_email)
    snippets = await sqlite_style_snippets(from_email, n=int(os.getenv("STYLE_SNIPPETS_N","4"))) if use_style else []
    style_seed = "\n\n".join(f"— {s}" for s in snippets) if snippets else ""

    if intent in ("status_request", "admin"):
        style += "\n" + customer_mail_guidance(lang, intent)

    evidence_policy = (
        "EVIDENCE POLICY\n"
        "• Recommend any intervention (IPR, distalization, elastics, TADs, intrusion, expansion) ONLY if the same term(s) appear in the 'Relevant context' AND the context supports its indication.\n"
        "• If context lacks such support, explicitly state what data is missing and avoid prescribing that step.\n"
        "• When key facts are missing (Angle class, Bolton, crowding/spacing in mm, OJ/OB, habits), ask for them under 'What we need' and defer specific plan details.\n"
        "• Use if/then decision rules. Prefer concise, checkable instructions with mm-values and review criteria.\n"
    )
    if facts.get("open_bite") or facts.get("deep_bite") or facts.get("crowding_mm") is None:
        evidence_policy += "• For space management (IPR/distal/expansion): require Bolton analysis or explicitly mark it as pending.\n"

    def _guess_greeting_target(txt: str) -> str:
        t = (txt or "").lower().strip()
        if t.startswith(("hej helle", "dear helle", "hej, helle")):
            return "to_helle"
        return "unknown"

    greet_mode = _guess_greeting_target(user_text)

    final_prompt = (
        f"{system_prompt}\n\n"
        "RANKED CONTEXT POLICY\n"
        "• HistoryDB (same clinic/ticket or high semantic match) = primary evidence.\n"
        "• Q&A evidence = primary when HistoryDB is absent or weak.\n"
        "• MAO (by anchors) = authoritative reference; if conflict with HistoryDB, prefer MAO unless HistoryDB documents a clinic-approved protocol.\n"
        "• MAO (keyword) = secondary fallback.\n"
        "• History (tone-only) = for stylistic alignment if not a strong match.\n\n"
        f"{policy_block}\n\n"
        f"{evidence_policy}\n"
        f"{style}\n"
        + (f"\nSTYLE SNIPPETS (tidligere mails til samme kunde):\n{style_seed}\n" if style_seed else "")
        + "\nIMPORTANT: Use only the information from 'Relevant context' below. Do not invent names, case numbers or internal details that are not present in the context.\n\n"
        f"Known brand/case (if any): {brand_case}\n\n"
        f"User message:\n{user_text}\n\n"
        f"Relevant context (may be in English and may be partial):\n{context}\n\n"
        f"Answer in the user's language (detected: {lang}, role: {role_disp}):"
    )
    logger.info(f"final_prompt(sample 400): {final_prompt[:400]}")

    try:
        answer_markdown = await get_rag_answer(final_prompt)
    except Exception as e:
        logger.exception(f"OpenAI call failed: {e}")
        answer_markdown = (
            "Beklager, der opstod en fejl under genereringen af svaret." if lang == "da"
            else "Sorry, an error occurred while generating the answer."
        )

    answer_plain = md_to_plain(answer_markdown)
    answer_out = answer_plain if output_mode in ("plain", "tech_brief", "mail") else answer_markdown

    sources = []
    for ch in combined_chunks[:6]:
        meta = ch.get("meta", {}) if isinstance(ch, dict) else {}
        label = _label_from_meta(meta)
        url = meta.get("url") if isinstance(meta, dict) else None
        src = {"label": label, "url": url}
        if isinstance(meta, dict) and meta.get("refs"):
            src["refs"] = meta.get("refs")
        if isinstance(meta, dict) and (meta.get("anchor_id") or meta.get("anchorId")):
            src["anchor_id"] = meta.get("anchor_id") or meta.get("anchorId")
        sources.append(src)

    # Normaliser hilsen, hvis kunden skriver "Hej Helle" men vi svarer generisk
    if lang == "da" and output_mode in ("mail", "plain"):
        low = answer_out.strip().lower()
        trimmed = answer_out.strip()
        if greet_mode == "to_helle":
            if low.startswith("hej helle"):
                cut = "hej helle,"
                if trimmed[:len(cut)].lower() == cut:
                    answer_out = "Hej,\n\n" + trimmed[len(cut):].lstrip()
                else:
                    cut2 = "hej helle"
                    answer_out = "Hej,\n\n" + trimmed[len(cut2):].lstrip()
        else:
            # Hvis vi har et kundenavn-hint, så læg en venlig hilsen på
            answer_out = _inject_greeting(answer_out, customer_name_hint, "da")

        if os.getenv("MAIL_HIDE_SOURCES", "1") == "1" and output_mode == "mail":
            sources = []

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

# Legacy shim
@app.post("/answer", dependencies=[Depends(require_rag_token)])
async def legacy_answer_proxy(request: Request):
    return await api_answer(request)

@app.post("/update_ticket")
async def noop_update_ticket():
    return {"status": "noop"}


# =========================
# [Y] VIGTIGT: Importér routers TIL SIDST
# =========================
from app.routers import admin_sync, chat
app.include_router(admin_sync.router)  # prefix defineres inde i router-filerne
app.include_router(chat.router)
from app.routers import admin_sync, chat, moderation
app.include_router(admin_sync.router)
app.include_router(chat.router)
app.include_router(moderation.router)
app.include_router(mod.router)
