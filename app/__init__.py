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

# -------- Metadata (multi-file support) --------
METADATA_FILE       = os.getenv("METADATA_FILE", "metadata.json")    # fallback (single file)
# Brug KUN METADATA_FILES nedenfor (kommasepareret/whitespace/semikolon)
METADATA_FILES_RAW  = os.getenv("METADATA_FILES", "")

LOCAL_DB_PATH      = os.getenv("LOCAL_DB_PATH", "/mnt/data/knowledge.sqlite")

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

# ===== Auto-translate (NYT) =====
AUTO_TRANSLATE_OUTPUT = os.getenv("AUTO_TRANSLATE_OUTPUT", "1") == "1"
_AUTO_LANGS_RAW = os.getenv("AUTO_TRANSLATE_LANGS", "da,de,fr")
AUTO_TRANSLATE_LANGS = {s.strip().lower() for s in _AUTO_LANGS_RAW.split(",") if s.strip()}

# Globals
_FAISS_INDEX = None
_METADATA: List[Dict[str, Any]] = []
_SQLITE_OK = False

# Q&A globals
_QA_ITEMS: List[Dict[str, Any]] = []

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

    # ensure dir for local db
    try:
        base = os.path.dirname(LOCAL_DB_PATH)
        if base:
            os.makedirs(base, exist_ok=True)
    except Exception:
        pass

    try:
        await download_db()
        await init_db()
        # --- load Q&A JSON (unchanged) ---
        global _QA_ITEMS
        _QA_ITEMS = _qa_load_items()
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
# Metadata loader (ROBUST, støtter 'chunks' mv. uden formatændring)
# =========================
def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def _json_to_list_any(txt: str) -> List[Dict[str, Any]]:
    """
    Accepterer:
      - JSON-liste
      - dict med nøgler: chunks/items/data/documents/rows -> liste
      - dict med liste-værdier -> flatten
      - NDJSON (én JSON pr. linje)
    Returnerer en liste af objekter; ikke-objekt-elementer filtreres bort.
    """
    import json
    def only_objs(seq):
        return [x for x in seq if isinstance(x, dict)]

    # 1) Prøv som 'almindelig' JSON
    try:
        obj = json.loads(txt)
        if isinstance(obj, list):
            return only_objs(obj)
        if isinstance(obj, dict):
            for k in ("chunks","items","data","documents","rows"):
                if isinstance(obj.get(k), list):
                    return only_objs(obj[k])
            # flatten liste-væier
            flat = []
            for v in obj.values():
                if isinstance(v, list):
                    flat.extend(v)
            if flat:
                return only_objs(flat)
    except Exception:
        pass

    # 2) NDJSON fallback
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

    # 3) Tomt
    return []

def _load_metadata_files() -> List[Dict[str, Any]]:
    paths_env = METADATA_FILES_RAW.strip()
    single = METADATA_FILE.strip()
    paths: List[str] = []

    if paths_env:
        # split på komma/semikolon/whitespace
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
# Init FAISS / metadata / SQLite (robust file loader)
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

    # METADATA (robust – understøtter din nuværende filstruktur)
    try:
        _METADATA = _load_metadata_files()
        if not _METADATA:
            logger.info("No metadata loaded (check METADATA_FILES or METADATA_FILE).")
    except Exception as e:
        logger.exception(f"Failed to load metadata: {e}")
        _METADATA = []

    # SQLite ping
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

# === NYT: Indlæs flere historiktråde som chunks ===
async def sqlite_load_threads_as_chunks(limit:int=2000) -> List[Dict[str,Any]]:
    """
    Læser 'tekstlige' kolonner på tværs af tabeller og laver generiske chunks.
    Returnerer items i form: {"text":..., "meta": {...}} der kan føjes til _METADATA.
    """
    if not _SQLITE_OK:
        return []
    out=[]
    try:
        async with aiosqlite.connect(LOCAL_DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            tables=[]
            async with db.execute("SELECT name FROM sqlite_master WHERE type='table'") as cur:
                async for row in cur: tables.append(row["name"])
            text_cols = ["plainText","plaintext","text","content","body","message","message_text"]
            id_cols   = ["ticketId","ticket_id","tid","id"]
            who_cols  = ["contactId","contact_id","from","sender","author"]
            time_cols = ["createdTime","createdAt","created_at","date","timestamp","time","updated_at"]

            for table in tables:
                try:
                    async with db.execute(f"PRAGMA table_info('{table}')") as cur:
                        cols=[dict(r) async for r in cur]
                    cn = {c["name"] for c in cols}
                    tcol = next((c for c in text_cols if c in cn), None)
                    if not tcol: 
                        continue
                    kcol = next((c for c in id_cols if c in cn), None)
                    wcol = next((c for c in who_cols if c in cn), None)
                    dcol = next((c for c in time_cols if c in cn), None)
                    sql = f"SELECT {tcol} AS txt, {kcol if kcol else 'NULL'} AS kid, {wcol if wcol else 'NULL'} AS who, {dcol if dcol else 'NULL'} AS dt FROM '{table}' ORDER BY rowid DESC LIMIT ?"
                    async with db.execute(sql, (limit,)) as cur:
                        async for r in cur:
                            txt=(r["txt"] or "").strip()
                            if not txt or len(txt)<40: 
                                continue
                            meta={"source":"HistoryDB","title":f"{table}",
                                  "ticketId": r["kid"], "contact": r["who"], "date": r["dt"]}
                            out.append({"text": txt, "meta": meta})
                except Exception:
                    continue
    except Exception as e:
        logger.info(f"sqlite_load_threads_as_chunks: {e}")
    return out

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

# === NYT: De-identifikation af PII i kontekst ===
def _deidentify(txt: str) -> str:
    if not txt:
        return txt
    s = txt
    s = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[email]", s)
    s = re.sub(r"\+?\d[\d\-\s]{6,}\d", "[phone]", s)
    s = re.sub(r"\b\d{6}\-?\d{4}\b", "[id]", s)
    s = re.sub(r"\b([A-ZÆØÅ][a-zæøå]{2,})(\s+[A-ZÆØÅ][a-zæøå]{2,}){0,2}\b", "[name]", s)
    return s

# ===== Klinisk faktekstraktor =====
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

# ===== Embeddings + semantic rerank + MMR =====
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
            scored.append((_cos(q_emb, emb), c))
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
        sims_q = [_cos(q_emb, e) for e in embs]
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

def style_hint(mode: str, lang: str) -> str:
    """
    Returnerer en streng, der instruerer modellen i præcis outputformatering.
    Understøtter: markdown | plain | tech_brief | mail
    """
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
            # Mail-skabelon på dansk — ingen 'Sources/Language' i brødteksten.
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
        return ""  # markdown (standard)
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
            # Mail-skabelon på engelsk
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
        return ""  # markdown (standard)

# =========================
# Q&A JSON search
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
# Retrieval helpers (generic metadata) + MAO helpers
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

async def translate_english_to(text: str, target_lang: str) -> str:
    """
    Oversæt fra engelsk til 'target_lang' uden ekstra forklaringer. Bevar linjeskift/markdown.
    """
    if not text or not target_lang or target_lang.lower() == "en":
        return text
    prompt = (
        f"Translate the following English text to {target_lang} only. "
        "Preserve the structure (line breaks, lists, headings) and do not add any explanations.\n\n"
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

    # 4) Cross-lingual retrieval + facts
    retrieval_query = await translate_to_english_if_needed(user_text, lang)
    if retrieval_query != user_text:
        logger.info("Query translated to English for retrieval.")
        logger.info(f"retrieval_query(sample 200): {retrieval_query[:200]}")

    facts = extract_facts(user_text)
    boosters = build_query_boosters(facts)
    boosted_query = (retrieval_query + " " + " ".join(boosters)).strip()

    # --- Q&A hits (stort K) ---
    qa_items = search_qa_json(boosted_query, k=RETR_K)
    qa_hits_raw = [_qa_to_chunk(x) for x in qa_items]

    # --- MAO via anchors fra Q&A refs ---
    qa_refs = [aid for it in qa_items for aid in (it.get("refs") or []) if isinstance(aid, str) and aid.startswith("MAO:")]
    mao_ref_hits_raw = find_mao_by_anchors(qa_refs, k_per_anchor=2) if qa_refs else []

    # --- metadata fallback (MAO keyword) ---
    mao_kw_hits_raw = []
    try:
        mao_kw_hits_raw = await get_mao_top_chunks(boosted_query, k=max(4, RETR_K//3))
    except Exception as e:
        logger.info(f"MAO keyword retrieval failed: {e}")

    # Merge kandidater, dedup, semantisk rerank, MMR-diversitet
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

    # --- historical tone/content (NYT: prioriter ved match) ---
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

    if history_primary:
        combined_chunks = history_primary + diversified
    else:
        combined_chunks = diversified + history_hits

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

    # Context text (NYT: de-identificér)
    def _clip(txt: str, max_len: int = 1200) -> str:
        return (txt[:max_len] + "…") if len(txt) > max_len else txt

    context = "\n\n".join(_clip(_deidentify(text_of(ch))) for ch in combined_chunks)[:9000]

    # 6) Compose final prompt (med evidens-policy)
    style = style_hint(output_mode, lang)

    evidence_policy = (
        "EVIDENCE POLICY\n"
        "• Recommend any intervention (IPR, distalization, elastics, TADs, intrusion, expansion) ONLY if the same term(s) appear in the 'Relevant context' AND the context supports its indication.\n"
        "• If context lacks such support, explicitly state what data is missing and avoid prescribing that step.\n"
        "• When key facts are missing (Angle class, Bolton, crowding/spacing in mm, OJ/OB, habits), ask for them under 'What we need' and defer specific plan details.\n"
        "• Use if/then decision rules. Prefer concise, checkable instructions with mm-values and review criteria.\n"
    )
    if facts.get("open_bite") or facts.get("deep_bite") or facts.get("crowding_mm") is None:
        evidence_policy += "• For space management (IPR/distal/expansion): require Bolton analysis or explicitly mark it as pending.\n"

    final_prompt = (
        f"{system_prompt}\n\n"
        "RANKED CONTEXT POLICY\n"
        "• HistoryDB (same clinic/ticket or high semantic match) = primary evidence.\n"
        "• Q&A evidence = primary when HistoryDB is absent or weak.\n"
        "• MAO (by anchors) = authoritative reference; if conflict with HistoryDB, prefer MAO unless HistoryDB documents a clinic-approved protocol.\n"
        "• MAO (keyword) = secondary fallback.\n"
        "• History (tone-only) = for stylistic alignment if not a strong match.\n\n"
        f"{evidence_policy}\n"
        f"{style}\n\n"
        "IMPORTANT: Use only the information from 'Relevant context' below. "
        "Do not invent names, case numbers or internal details that are not present in the context.\n\n"
        f"User message:\n{user_text}\n\n"
        f"Relevant context (may be in English and may be partial):\n{context}\n\n"
        "Answer in English only:"
    )
    logger.info(f"final_prompt(sample 400): {final_prompt[:400]}")

    # 7) LLM (always generate in English)
    try:
        answer_markdown = await get_rag_answer(final_prompt)
    except Exception as e:
        logger.exception(f"OpenAI call failed: {e}")
        answer_markdown = "Sorry, an error occurred while generating the answer."

    # 7b) Optional auto-translate back to user's language (skip for mail)
    if AUTO_TRANSLATE_OUTPUT and lang and lang != "en" and lang.lower() in AUTO_TRANSLATE_LANGS and output_mode != "mail":
        try:
            translated = await translate_english_to(answer_markdown, lang)
            if translated and translated.strip():
                answer_markdown = translated
                logger.info(f"Auto-translated output to '{lang}'.")
            else:
                logger.info("Auto-translate produced empty output; keeping English.")
        except Exception as e:
            logger.info(f"Auto-translate skipped due to error: {e}")

    # 8) Plain rendering decision AFTER possible translation
    answer_plain = md_to_plain(answer_markdown)
    if output_mode in ("plain", "tech_brief", "mail"):
        # 'mail' skal ALTID være ren tekst-epost på ENGELSK (vi oversatte ikke mail ovenfor)
        answer_out = answer_plain
    else:
        answer_out = answer_markdown

    # 9) Sources (byg direkte fra combined_chunks, så rækkefølge matcher)
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
