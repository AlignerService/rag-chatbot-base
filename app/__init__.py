# app/__init__.py
import os
import json
import logging
import asyncio
import hmac
import re
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np  # optional
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
LOCAL_DB_PATH      = os.getenv("LOCAL_DB_PATH", "/mnt/data/rag.sqlite3")

RAG_BEARER_TOKEN   = os.getenv("RAG_BEARER_TOKEN", "")

OPENAI_MODEL       = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "")

# -------- Dropbox creds (both modes supported) --------
DROPBOX_ACCESS_TOKEN  = os.getenv("DROPBOX_ACCESS_TOKEN", "")     # simple, short-lived token (or long-lived in dev)
DROPBOX_DB_PATH       = os.getenv("DROPBOX_DB_PATH", "")          # e.g. "/Apps/AlignerService/rag.sqlite3"

# Refresh-token flow (matches your Render variables)
DROPBOX_CLIENT_ID     = os.getenv("DROPBOX_CLIENT_ID", "")
DROPBOX_CLIENT_SECRET = os.getenv("DROPBOX_CLIENT_SECRET", "")
DROPBOX_REFRESH_TOKEN = os.getenv("DROPBOX_REFRESH_TOKEN", "")

# -------- Q&A JSON integration (env-config) --------
QA_JSON_PATH   = os.getenv("QA_JSON_PATH", "mao_qa_rag_export.json")
QA_JSON_ENABLE = os.getenv("QA_JSON_ENABLE", "1") == "1"

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
        await download_db()  # pulls SQLite from Dropbox (access token OR refresh flow)
        await init_db()      # loads FAISS, metadata.json, verifies SQLite
        # --- load Q&A JSON (new) ---
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
        "de": {
            "dentist": ["zahnarzt", "zahnärztin"],
            "orthodontist": ["kieferorthopäde", "kieferorthopädin", "kfo"],
            "assistant": ["zfa", "assistenz", "stuhlassistenz"],
            "hygienist": ["dentalhygieniker", "dentalhygienikerin", "dh"],
            "receptionist": ["rezeption", "empfang", "praxismanager"],
            "team": ["team", "praxisteam"],
        },
        "fr": {
            "dentist": ["dentiste", "chirurgien-dentiste"],
            "orthodontist": ["orthodontiste"],
            "assistant": ["assistante dentaire", "assistant dentaire"],
            "hygienist": ["hygiéniste dentaire", "hygieniste dentaire"],
            "receptionist": ["réceptionniste", "accueil", "secrétaire médicale"],
            "team": ["équipe", "cabinet", "clinique"],
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
        "de": {"dentist": "Zahnarzt/Zahnärztin","orthodontist":"Kieferorthopäde/Kieferorthopädin","assistant":"ZFA/Assistenz","hygienist":"Dentalhygieniker/in","receptionist":"Rezeption/PM","team":"Praxisteam","clinician":"Behandler/in"},
        "fr": {"dentist": "dentiste","orthodontist":"orthodontiste","assistant":"assistant(e) dentaire","hygienist":"hygiéniste dentaire","receptionist":"réceptionniste","team":"équipe clinique","clinician":"praticien(ne)"},
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
    role_text_de = {
        "dentist": "• Für Zahnärzt:innen: klinische Parameter, Planungsentscheidungen, Risiken/Kontraindikationen, Dokumentation.",
        "orthodontist": "• Für KFO: Staging, Attachments/Engager, IPR-Verteilung, Elastiks, Biomechanik.",
        "assistant": "• Für Assistenz/ZFA: Chairside-Checklisten, Dokumentation, Foto/Scan-Protokolle, Eskalation.",
        "hygienist": "• Für DH: Hygiene/Compliance, Instruktion, Beobachtungen; keine Planänderung.",
        "receptionist": "• Für Rezeption/PM: Vorlagen, (Um)Terminierung, Vorbereitung; keine klinischen Ratschläge.",
        "team": "• Für Team: Übergaben, Checklisten, Aufgabenverteilung; Entscheidungen bei ZA/KFO.",
        "clinician": "• Für Behandler:innen: klinische Parameter, Protokolle, Entscheidungsregeln.",
    }
    role_text_fr = {
        "dentist": "• Pour dentistes : paramètres cliniques, choix de planification, risques/contre-indications, documentation.",
        "orthodontist": "• Pour orthodontistes : staging, attachments/engagers, répartition IPR, élastiques, biomécanique.",
        "assistant": "• Pour assistant(e)s : check-lists au fauteuil, champs de doc, protocoles photo/scan, critères d’escalade.",
        "hygienist": "• Pour hygiénistes : hygiène/compliance, éducation, observations ; pas de modification du plan.",
        "receptionist": "• Pour accueil : modèles, (re)prise de RDV, liste de préparation ; pas de conseil clinique.",
        "team": "• Pour équipe : handoffs, check-lists, répartition des tâches ; décisions chez dentiste/orthodontiste.",
        "clinician": "• Pour praticien(ne)s : paramètres cliniques, protocoles, règles décisionnelles.",
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
    role_map = {"da": role_text_da, "de": role_text_de, "fr": role_text_fr, "en": role_text_en}
    role_line = role_map.get(lang, role_text_en).get(role, role_map.get(lang, role_text_en)["clinician"])

    if lang == "da":
        return (
            "Du er AI-assistenten for tandlæge **Helle Hatt** (ekspert i clear aligners). "
            "Du svarer KUN til professionelle (tandlæger, ortodontister, klinikteams) — aldrig patienter.\n\n"
            "KILDER: Brug PRIMÆRT 'Relevant context' (SQLite/Dropbox, bog, blog). Hvis utilstrækkelig: sig det og giv kun etablerede best practices — "
            "opfind aldrig politikker, sagsnumre, navne eller data uden for konteksten. Konteksten kan være engelsk; oversæt terminologi naturligt til dansk.\n\n"
            f"ROLLEFOKUS\n{role_line}\n\n"
            "FORMAT\n• Kort konklusion (1–2 sætninger)\n• Struktureret protokol (nummererede trin med kliniske parametre: mm IPR, 22 t/d, staging)\n"
            "• Beslutningsregler (if/then) + risici/kontraindikationer\n• Næste skridt (2–4 punkter) + evt. journal-/opgavenote\n\n"
            "SIKKERHED\n• Ingen patient-specifik diagnose/ordination uden tilstrækkelig info; anfør usikkerheder kort. "
            "• Ingen direkte patienthenvendelse. Opret aldrig interne detaljer, der ikke er i konteksten."
        )
    if lang == "de":
        return (
            "Du bist die KI-Assistenz von **Dr. Helle Hatt** (Clear-Aligner-Expertin). "
            "Du adressierst AUSSCHLIESSLICH Fachleute.\n\n"
            "QUELLEN: Primär 'Relevant context'. Bei Lücken: offen sagen und nur etablierte Best Practices liefern — "
            "nichts erfinden (Richtlinien, Fallnummern, Namen, Daten). Kontext ggf. auf Englisch; Terminologie natürlich auf Deutsch.\n\n"
            f"ROLLE\n{role_line}\n\n"
            "FORMAT\n• Kurze Zusammenfassung (1–2 Sätze)\n• Strukturiertes Protokoll (mm IPR, 22 h/Tag, Staging)\n"
            "• Entscheidungsregeln + Risiken/Kontraindikationen\n• Nächste Schritte (2–4) + ggf. Journal-/Aufgaben-Notiz\n\n"
            "SICHERHEIT\n• Keine patientenspez. Diagnose/Anordnung ohne ausreichende Info; Unsicherheiten kurz nennen. "
            "• Keine Patientenansprache. Keine erfundenen internen Details."
        )
    if lang == "fr":
        return (
            "Vous êtes l’assistant IA de **la Dr Helle Hatt** (experte en aligneurs). "
            "Vous vous adressez EXCLUSIVEMENT aux professionnels.\n\n"
            "SOURCES : Priorité au « Relevant context ». Si insuffisant : le dire et fournir uniquement des bonnes pratiques établies — "
            "ne rien inventer (politiques, n° de dossier, noms, données). Contexte possiblement en anglais ; traduire naturellement en français.\n\n"
            f"FOCUS RÔLE\n{role_line}\n\n"
            "FORMAT\n• Conclusion brève (1–2 phrases)\n• Protocole structuré (mm d’IPR, 22 h/j, staging)\n"
            "• Règles décisionnelles + risques/contre-indications\n• Étapes suivantes (2–4) + note dossier/tâche\n\n"
            "SÉCURITÉ\n• Pas de diagnostic/ordonnance spécifique sans données suffisantes ; incertitudes brèves. "
            "• Pas d’adresse patient. Ne pas inventer de détails internes."
        )
    return (
        "You are the AI assistant for **Dr. Helle Hatt** (clear-aligner expert). "
        "Address PROFESSIONALS ONLY.\n\n"
        "SOURCES: Rely on 'Relevant context'. If insufficient: state it and provide only established best practices — "
        "never invent policies, case numbers, names or data. Context may be in English; translate naturally.\n\n"
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
    """
    Download SQLite from Dropbox to LOCAL_DB_PATH.
    Supports:
      - DROPBOX_ACCESS_TOKEN (simple)
      - OR refresh flow with DROPBOX_CLIENT_ID + DROPBOX_CLIENT_SECRET + DROPBOX_REFRESH_TOKEN
    Requires DROPBOX_DB_PATH (e.g. "/Apps/AlignerService/rag.sqlite3" or "/rag.sqlite3" for App Folder apps).
    """
    if not DROPBOX_DB_PATH:
        logger.info("download_db(): no-op (DROPBOX_DB_PATH not set)")
        return

    try:
        import dropbox  # type: ignore
    except Exception as e:
        logger.warning(f"Dropbox SDK not available ({e}); skipping DB download.")
        return

    try:
        # Choose auth mode
        if DROPBOX_ACCESS_TOKEN:
            dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)
        elif DROPBOX_REFRESH_TOKEN and DROPBOX_CLIENT_ID and DROPBOX_CLIENT_SECRET:
            dbx = dropbox.Dropbox(
                oauth2_refresh_token=DROPBOX_REFRESH_TOKEN,
                app_key=DROPBOX_CLIENT_ID,
                app_secret=DROPBOX_CLIENT_SECRET,
            )
        else:
            logger.warning("Dropbox credentials missing: set either DROPBOX_ACCESS_TOKEN or (DROPBOX_CLIENT_ID, DROPBOX_CLIENT_SECRET, DROPBOX_REFRESH_TOKEN).")
            return

        # Normalize path
        dbx_path = DROPBOX_DB_PATH if DROPBOX_DB_PATH.startswith("/") else f"/{DROPBOX_DB_PATH}"

        # Ensure local dir exists
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
# SQLite helpers (tolerant schema)
# =========================
async def sqlite_latest_thread_plaintext(ticket_id: str, contact_id: Optional[str] = None) -> str:
    """
    Try to find the latest message text for a given ticket_id (and optional contact_id)
    by scanning tables/columns heuristically. Returns plain text or "" if not found.
    """
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
            # list tables
            tables = []
            async with db.execute("SELECT name FROM sqlite_master WHERE type='table'") as cur:
                async for row in cur:
                    tables.append(row["name"])

            best_text = ""
            best_time = None

            for table in tables:
                # columns
                async with db.execute(f"PRAGMA table_info('{table}')") as cur:
                    cols = [dict(row) async for row in cur]
                col_names = [c["name"] for c in cols]

                text_cols = [c for c in candidates_text if c in col_names]
                ticket_cols = [c for c in candidates_ticket if c in col_names]
                contact_cols = [c for c in candidates_contact if c in col_names]
                time_cols = [c for c in candidates_time if c in col_names]

                if not text_cols or not ticket_cols:
                    continue  # cannot filter by ticket or no text

                # Prefer the first candidate in each list
                tcol = text_cols[0]
                kcol = ticket_cols[0]
                ccol = contact_cols[0] if contact_cols else None
                timecol = time_cols[0] if time_cols else None

                # Build SQL
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
                                # choose the most recent across tables
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

def search_qa_json(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Returnerer top-k Q&A-objekter (uændret struktur) fra mao_qa_rag_export.json."""
    if not _QA_ITEMS:
        return []
    qtok = _qa_tokenize(query)
    scored = []
    for it in _QA_ITEMS:
        hay = " ".join(
            [it.get("question", "")]
            + (it.get("synonyms", []) or [])
            + [it.get("answer_markdown", "")]
        )
        stokens = _qa_tokenize(hay)
        s = _qa_score(qtok, stokens)
        if s > 0:
            scored.append((s, it))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [it for _, it in scored[:k]]

def _qa_to_chunk(it: Dict[str, Any]) -> Dict[str, Any]:
    # Formateres som et "chunk" så resten af din pipeline kan bruge det direkte
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
            temperature=0.0,
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
            temperature=0.0,
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
            temperature=0.2,
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
            temperature=0.2,
        )
        return (resp["choices"][0]["message"]["content"] or "").strip()
    except Exception as e_legacy:
        logger.exception(f"OpenAI call failed: {e_legacy}")
        return "I could not generate a response due to an internal error."

# =========================
# Retrieval helpers
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
        text = _extract_text_from_meta(item)
        text = _strip_html(text)
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
# Endpoints
# =========================
@app.get("/")
async def health():
    return {"status": "ok"}

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

        # classic fields (optional)
        if not user_text:
            for key_guess in ["plainText", "text", "message", "content", "body"]:
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

    # --- NEW: Q&A hits (from JSON) ---
    qa_hits = [ _qa_to_chunk(x) for x in search_qa_json(retrieval_query, k=3) ]

    try:
        top_chunks = await get_top_chunks(retrieval_query)
    except Exception as e:
        logger.exception(f"get_top_chunks failed: {e}")
        top_chunks = []

    # 5) Failsafe if no context at all
    if not qa_hits and not top_chunks:
        safe_generic = {
            "da": (
                "Grundprotokol når kontekst mangler:\n"
                "1) Screening/diagnose → 2) Plan (staging, IPR, attachments) → 3) Start → 4) Kontroller/tracking → 5) Refinement → 6) Retention.\n"
                "Næste skridt: indlæs flere kildedata i RAG for mere målrettet svar."
            ),
            "de": (
                "Basisprotokoll bei fehlendem Kontext:\n"
                "1) Screening/Diagnose → 2) Planung (Staging, IPR, Attachments) → 3) Start → 4) Kontrollen/Tracking → 5) Refinement → 6) Retention.\n"
                "Nächste Schritte: mehr Quelldaten ins RAG laden."
            ),
            "fr": (
                "Protocole de base en l’absence de contexte :\n"
                "1) Dépistage/diagnostic → 2) Planification (staging, IPR, attachments) → 3) Démarrage → 4) Contrôles/tracking → 5) Refinement → 6) Rétention.\n"
                "Étapes suivantes : charger davantage de sources dans le RAG."
            ),
            "en": (
                "Baseline protocol when context is missing:\n"
                "1) Screening/diagnosis → 2) Planning (staging, IPR, attachments) → 3) Start → 4) Reviews/tracking → 5) Refinement → 6) Retention.\n"
                "Next steps: load more sources into RAG."
            ),
        }
        sources = []
        return {
            "finalAnswer": safe_generic.get(lang, safe_generic["en"]),
            "language": lang,
            "role": role_disp,
            "sources": sources,
            "ticketId": ticket_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            # Zoho-friendly alias:
            "message": {"content": safe_generic.get(lang, safe_generic["en"]), "language": lang}
        }

    # Combine Q&A chunks first, then metadata chunks
    combined_chunks = qa_hits + top_chunks

    context = "\n\n".join(ch.get("text", "") if isinstance(ch, dict) else str(ch) for ch in combined_chunks)[:8000]

    # 6) Compose final prompt
    final_prompt = (
        f"{system_prompt}\n\n"
        f"IMPORTANT: Use only the information from 'Relevant context' below. "
        f"Do not invent names, case numbers or internal details that are not present in the context.\n\n"
        f"User message:\n{user_text}\n\n"
        f"Relevant context (may be in English and may be partial):\n{context}\n\n"
        f"Answer in the user's language (detected: {lang}, role: {role_disp}):"
    )
    logger.info(f"final_prompt(sample 400): {final_prompt[:400]}")

    # 7) LLM
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

    # 8) Sources (show top of each group)
    sources = []
    for ch in (qa_hits[:2] + top_chunks[:2]):
        meta = ch.get("meta", {})
        label = _label_from_meta(meta)
        url = meta.get("url") if isinstance(meta, dict) else None
        # include refs if present (useful for audit)
        refs = meta.get("refs") if isinstance(meta, dict) else None
        src = {"label": label, "url": url}
        if refs:
            src["refs"] = refs
        sources.append(src)

    return {
        "finalAnswer": answer,
        "language": lang,
        "role": role_disp,
        "sources": sources,
        "ticketId": ticket_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        # Zoho-friendly alias
        "message": {"content": answer, "language": lang}
    }
