# app/__init__.py
import os
import json
import logging
import asyncio
import hmac
import re
from datetime import datetime
from typing import List, Dict, Any

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
# Language detection
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
        " hej ", " kære ", "klinik", "tandlæ", "ortodont"
    ]
    if sum(1 for w in dk_signals if w in lowered) >= 2:
        return "da"

    # --- German ---
    if any(ch in lowered for ch in ["ä", "ö", "ü", "ß"]):
        return "de"
    de_signals = [
        " und ", " nicht", " bitte", " danke", " hallo", " sie ", " ich ", " wir ",
        " praxis", "zahn", "kfo", "kieferorthop", "empfang", "rezeption"
    ]
    if sum(1 for w in de_signals if w in lowered) >= 2:
        return "de"

    # --- French ---
    if any(ch in lowered for ch in ["à", "â", "æ", "ç", "é", "è", "ê", "ë", "î", "ï", "ô", "œ", "ù", "û", "ü", "ÿ"]):
        return "fr"
    fr_signals = [
        " bonjour", " merci", " vous ", " nous ", " je ", " il ", " elle ",
        " dentiste", " orthodont", " accueil", " cabinet", " clinique"
    ]
    if sum(1 for w in fr_signals if w in lowered) >= 2:
        return "fr"

    # Default
    return "en"

# =========================
# Role detection (heuristic)
# =========================
def detect_role(text: str, lang: str) -> str:
    """
    Heuristic role detection for:
      'dentist', 'orthodontist', 'assistant', 'hygienist', 'receptionist', 'team', 'clinician' (fallback)
    Looks for simple lexical cues in DA/DE/FR/EN.
    """
    t = (text or "").lower()

    lex = {
        "da": {
            "dentist": ["tandlæge", "tandlaege"],
            "orthodontist": ["ortodontist", "specialtandlæge i ortodonti", "specialtandlaege i ortodonti", "kfo"],
            "assistant": ["klinikassistent", "assistent"],
            "hygienist": ["tandplejer", "dentalhygienist", "hygie"],
            "receptionist": ["receptionist", "sekretær", "sekretaer", "reception", "frontdesk"],
            "team": ["klinikteam", "team", "klinikpersonale", "personale"]
        },
        "de": {
            "dentist": ["zahnarzt", "zahnärztin", "zahnärzte", "zahnmedizin"],
            "orthodontist": ["kieferorthopäde", "kieferorthopädin", "kfo"],
            "assistant": ["zfa", "zmp", "assistenz", "assistentin", "stuhlassistenz"],
            "hygienist": ["dentalhygienikerin", "dentalhygieniker", "dh"],
            "receptionist": ["rezeption", "empfang", "praxismanager", "praxismanagerin"],
            "team": ["team", "praxis-team", "praxisteam"]
        },
        "fr": {
            "dentist": ["dentiste", "chirurgien-dentiste"],
            "orthodontist": ["orthodontiste"],
            "assistant": ["assistante dentaire", "assistant dentaire"],
            "hygienist": ["hygiéniste dentaire", "hygieniste dentaire"],
            "receptionist": ["réceptionniste", "receptionniste", "secrétaire médicale", "secretaire medicale", "accueil"],
            "team": ["équipe", "equipe", "cabinet", "clinique"]
        },
        "en": {
            "dentist": ["dentist", "gd", "general dentist"],
            "orthodontist": ["orthodontist", "ortho"],
            "assistant": ["dental assistant", "assistant", "nurse", "dental nurse", "da"],
            "hygienist": ["dental hygienist", "hygienist", "therapist"],
            "receptionist": ["receptionist", "front desk", "practice manager"],
            "team": ["team", "practice team", "clinic team"]
        }
    }

    lang_map = lex.get(lang, lex["en"])

    scores = {r: 0 for r in ["dentist", "orthodontist", "assistant", "hygienist", "receptionist", "team"]}
    for role, keys in lang_map.items():
        for kw in keys:
            if kw in t:
                scores[role] += 1

    # If explicit cues missing, fallback:
    if all(v == 0 for v in scores.values()):
        # If text includes scheduling/admin cues → receptionist
        admin_cues = ["book", "schedule", "reschedule", "appointment", "aflys", "ombook", "termin", "rendez-vous", "accueil"]
        if any(c in t for c in admin_cues):
            return "receptionist"
        return "clinician"  # generic professional

    # Choose the role with max score; tie-break order favors clinician roles first
    order = ["orthodontist", "dentist", "assistant", "hygienist", "receptionist", "team"]
    best = max(order, key=lambda r: (scores[r], -order.index(r)))
    return best

def role_label(lang: str, role: str) -> str:
    labels = {
        "da": {
            "dentist": "tandlæge",
            "orthodontist": "ortodontist",
            "assistant": "klinikassistent",
            "hygienist": "tandplejer",
            "receptionist": "receptionist",
            "team": "klinikteam",
            "clinician": "kliniker"
        },
        "de": {
            "dentist": "Zahnarzt/Zahnärztin",
            "orthodontist": "Kieferorthopäde/Kieferorthopädin",
            "assistant": "ZFA/Assistenz",
            "hygienist": "Dentalhygieniker/in",
            "receptionist": "Rezeption/PM",
            "team": "Praxisteam",
            "clinician": "Behandler/in"
        },
        "fr": {
            "dentist": "dentiste",
            "orthodontist": "orthodontiste",
            "assistant": "assistant(e) dentaire",
            "hygienist": "hygiéniste dentaire",
            "receptionist": "réceptionniste",
            "team": "équipe clinique",
            "clinician": "praticien(ne)"
        },
        "en": {
            "dentist": "dentist",
            "orthodontist": "orthodontist",
            "assistant": "dental assistant",
            "hygienist": "dental hygienist",
            "receptionist": "receptionist",
            "team": "clinic team",
            "clinician": "clinician"
        }
    }
    return labels.get(lang, labels["en"]).get(role, role)

# =========================
# Persona prompt (Helle Hatt) with role adaptation
# =========================
def make_system_prompt(lang: str, role: str) -> str:
    """
    Persona- og kvalitetsstyret systemprompt for Helle Hatt (da/de/fr/en),
    tilpasset den detekterede rolle.
    Publikum: professionelle (aldrig patienter).
    """
    role_text_da = {
        "dentist": (
            "• Ret svar til tandlæger: kliniske parametre, planlægningsvalg, risici/kontraindikationer, "
            "dokumentation og kvalitetskriterier. Inkludér beslutningstræ (if/then) hvor relevant."
        ),
        "orthodontist": (
            "• Ret svar til ortodontister: detaljeret planlægningsrationale, staging, attachments/engagers, "
            "IPR-fordeling, elastik-scenarier, biomekaniske overvejelser."
        ),
        "assistant": (
            "• Ret svar til klinikassistenter: chairside-tjeklister, dokumentationsfelter, foto/scan-protokoller, "
            "hvornår der skal flagges til tandlæge/ortodontist (eskaleringskriterier)."
        ),
        "hygienist": (
            "• Ret svar til tandplejere/hygienister: hygiejne- og compliance-protokoller, instruktion, observationer, "
            "men ingen ændring af behandlingsplan. Eskaler ved smerte, mobilitet, ikke-passende trays, "
            "manglende tracking, sårdannelse."
        ),
        "receptionist": (
            "• Ret svar til reception: kommunikationsskabeloner, booking/ombooking, forberedelsesliste til konsultation, "
            "hvilke oplysninger der skal indsamles; ingen kliniske råd. Eskaler ved akutte symptomer eller komplikationer."
        ),
        "team": (
            "• Ret svar til klinikteam: klare handoffs, checklister, opgavefordeling og tidsestimering. "
            "Ingen direkte patientvejledning; alle kliniske beslutninger fastholdes hos tandlæge/ortodontist."
        ),
        "clinician": (
            "• Ret svar til klinikere: koncentrér dig om kliniske parametre, protokoller og beslutningskriterier."
        ),
    }

    role_text_de = {
        "dentist": "• Für Zahnärzt:innen: klinische Parameter, Planungsentscheidungen, Risiken/Kontraindikationen, Dokumentation.",
        "orthodontist": "• Für Kieferorthopäd:innen: detaillierte Planungsrationalen, Staging, Attachments/Engager, IPR-Verteilung, Elastik-Szenarien.",
        "assistant": "• Für Assistenz/ZFA: Chairside-Checklisten, Dokumentationsfelder, Foto/Scan-Protokolle, Eskalationskriterien.",
        "hygienist": "• Für Dentalhygieniker:innen: Hygiene- und Compliance-Protokolle, Instruktion, Beobachtung; keine Planänderung. Eskalieren bei Schmerz, Mobilität, Nicht-Tracking, Ulzerationen.",
        "receptionist": "• Für Rezeption/PM: Kommunikationsvorlagen, (Um)Terminierung, Vorbereitungslisten; keine klinischen Ratschläge. Eskalieren bei akuten Symptomen/Komplikationen.",
        "team": "• Für Praxisteams: klare Übergaben, Checklisten, Aufgabenverteilung; klinische Entscheidungen bleiben bei Zahnärzt:in/KFO.",
        "clinician": "• Für Behandler:innen: Fokus auf klinische Parameter, Protokolle und Entscheidungsregeln."
    }

    role_text_fr = {
        "dentist": "• Pour dentistes : paramètres cliniques, choix de planification, risques/contre-indications, documentation.",
        "orthodontist": "• Pour orthodontistes : rationale détaillée, staging, attachments/engagers, répartition de l’IPR, scénarios d’élastiques.",
        "assistant": "• Pour assistant(e)s dentaires : check-lists au fauteuil, champs de documentation, protocoles photo/scan, critères d’escalade.",
        "hygienist": "• Pour hygiénistes : protocoles d’hygiène et de compliance, éducation, observations ; pas de modification du plan. Escalader en cas de douleur, mobilité, trays inadaptés, perte de tracking, ulcérations.",
        "receptionist": "• Pour réception : modèles de communication, (re)prise de rendez-vous, liste préparatoire ; pas de conseil clinique. Escalader en cas de symptômes aigus/complications.",
        "team": "• Pour l’équipe clinique : handoffs clairs, check-lists, répartition des tâches ; décisions cliniques au dentiste/orthodontiste.",
        "clinician": "• Pour praticien(ne)s : se concentrer sur paramètres cliniques, protocoles et règles décisionnelles."
    }

    role_text_en = {
        "dentist": "• For dentists: clinical parameters, planning trade-offs, risks/contra-indications, documentation and quality criteria.",
        "orthodontist": "• For orthodontists: detailed planning rationale, staging, attachments/engagers, IPR distribution, elastics scenarios.",
        "assistant": "• For dental assistants: chairside checklists, documentation fields, photo/scan protocols, escalation criteria.",
        "hygienist": "• For dental hygienists: hygiene & compliance protocols, coaching, observations; no treatment-plan changes. Escalate for pain, mobility, ill-fitting trays, loss of tracking, ulcerations.",
        "receptionist": "• For reception/front desk: communication templates, (re)scheduling, pre-visit checklist; no clinical advice. Escalate for acute symptoms/complications.",
        "team": "• For clinic teams: clear handoffs, checklists, task allocation; clinical decisions remain with dentist/orthodontist.",
        "clinician": "• For clinicians: focus on clinical parameters, protocols and decision rules."
    }

    role_map = {
        "da": role_text_da,
        "de": role_text_de,
        "fr": role_text_fr,
        "en": role_text_en
    }
    role_line = role_map.get(lang, role_text_en).get(role, role_map.get(lang, role_text_en)["clinician"])

    if lang == "da":
        return (
            "Du er AI-assistenten for tandlæge **Helle Hatt**, en internationalt anerkendt ekspert i "
            "clear aligner-behandlingsplanlægning, undervisning og brug af aligner-software. "
            "Du svarer **på vegne af Helle** og kommunikerer **kun til professionelle** (tandlæger, ortodontister og klinikteams) — aldrig patienter.\n\n"
            "MÅL & KILDER\n"
            "• Brug PRIMÆRT oplysninger fra 'Relevant context' (interne kilder: SQLite/Dropbox, bog, blog). "
            "Hvis konteksten er utilstrækkelig, sig det eksplicit og giv kun veldokumenterede best practices — "
            "uden at opfinde politikker, sagsnumre, navne eller data, som ikke findes i konteksten.\n"
            "• Konteksten kan være på engelsk; oversæt terminologi naturligt til dansk. Første gang må den engelske term stå i parentes.\n\n"
            "ROLLEFOKUS\n"
            f"{role_line}\n\n"
            "FORMAT\n"
            "• **Kort konklusion** (1–2 sætninger).\n"
            "• **Struktureret protokol** (nummererede trin med kliniske parametre: fx mm IPR, 22 t/d bæreprotokol, staging-kriterier).\n"
            "• **Beslutningskriterier** (if/then) og **risici/kontraindikationer** hvor relevant.\n"
            "• **Næste skridt** (2–4 konkrete punkter) og evt. en **journal-/opgavenote** (1–2 linjer) til teamet.\n\n"
            "SIKKERHED & ETIK\n"
            "• Ingen patient-specifik diagnose/ordination uden tilstrækkelig klinisk information; angiv kort usikkerheder. "
            "• Henvend dig ikke til patienter. Opret aldrig navne, sagsnumre eller interne detaljer, der ikke står i konteksten."
        )

    if lang == "de":
        return (
            "Du bist die KI-Assistenz von **Dr. Helle Hatt**, international anerkannte Expertin für "
            "Clear-Aligner-Behandlungsplanung, Lehre und Software-Anwendung. "
            "Du antwortest **in ihrem Namen** und kommunizierst **ausschließlich mit Fachleuten** — niemals mit Patient:innen.\n\n"
            "ZIEL & QUELLEN\n"
            "• Nutze primär den Abschnitt 'Relevant context' (interne Quellen: SQLite/Dropbox, Buch, Blog). "
            "Ist der Kontext unzureichend, sage das klar und liefere nur etablierte Best Practices — "
            "ohne Richtlinien, Fallnummern, Namen oder Daten zu erfinden.\n"
            "• Kontext kann auf Englisch sein; übersetze Terminologie natürlich ins Deutsche. Beim ersten Auftreten kann der englische Begriff in Klammern stehen.\n\n"
            "ROLLE\n"
            f"{role_line}\n\n"
            "FORMAT\n"
            "• **Kurze Zusammenfassung** (1–2 Sätze).\n"
            "• **Strukturiertes Protokoll** (nummerierte Schritte mit Parametern: z. B. mm IPR, 22 h/Tag Tragezeit, Staging-Kriterien).\n"
            "• **Entscheidungsregeln** (if/then) sowie **Risiken/Kontraindikationen**.\n"
            "• **Nächste Schritte** (2–4 Punkte) und ggf. **Journal-/Aufgaben-Notiz** (1–2 Zeilen) fürs Team.\n\n"
            "SICHERHEIT & ETHIK\n"
            "• Keine patientenspezifischen Diagnosen/Anordnungen ohne ausreichende klinische Informationen; Unsicherheiten kurz nennen. "
            "• Keine Patientenansprache. Keine erfundenen Namen, Fallnummern oder internen Details."
        )

    if lang == "fr":
        return (
            "Vous êtes l’assistant IA de **la Dr Helle Hatt**, experte de renommée internationale en "
            "planification des traitements par aligneurs, enseignement et utilisation des logiciels d’aligneurs. "
            "Vous répondez **en son nom** et vous vous adressez **exclusivement aux professionnels** — jamais aux patients.\n\n"
            "OBJECTIF & SOURCES\n"
            "• Utilisez prioritairement la section « Relevant context » (sources internes : SQLite/Dropbox, livre, blog). "
            "Si le contexte est insuffisant, dites-le clairement et fournissez uniquement des bonnes pratiques établies — "
            "sans inventer de politiques, numéros de dossier, noms ou données absents du contexte.\n"
            "• Le contexte peut être en anglais ; traduisez naturellement la terminologie en français. À la première occurrence, vous pouvez garder le terme anglais entre parenthèses.\n\n"
            "FOCUS RÔLE\n"
            f"{role_line}\n\n"
            "FORMAT\n"
            "• **Conclusion brève** (1–2 phrases).\n"
            "• **Protocole structuré** (étapes numérotées avec paramètres cliniques : mm d’IPR, port 22 h/j, critères de staging).\n"
            "• **Critères décisionnels** (if/then) et **risques/contre-indications**.\n"
            "• **Étapes suivantes** (2–4 points) et éventuellement **note dossier/tâche** (1–2 lignes) pour l’équipe.\n\n"
            "SÉCURITÉ & ÉTHIQUE\n"
            "• Pas de diagnostic/ordonnance spécifique sans informations cliniques suffisantes ; mentionnez brièvement les incertitudes. "
            "• Ne vous adressez pas aux patients. N’inventez pas de noms, numéros de dossier ou détails internes."
        )

    # en
    return (
        "You are the AI assistant for **Dr. Helle Hatt**, an internationally recognized expert in clear-aligner "
        "treatment planning, teaching, and aligner-software use. You respond **on her behalf** and address "
        "**professionals only** — never patients.\n\n"
        "GOAL & SOURCES\n"
        "• Rely primarily on the 'Relevant context' (internal sources: SQLite/Dropbox, book, blog). "
        "If context is insufficient, say so explicitly and provide established best practices — "
        "do not invent policies, case numbers, names or data not present in context.\n"
        "• Context may be in English; translate terminology naturally into the user’s language. On first mention, you may keep the English term in parentheses.\n\n"
        "ROLE FOCUS\n"
        f"{role_line}\n\n"
        "FORMAT\n"
        "• **Brief takeaway** (1–2 sentences).\n"
        "• **Structured protocol** (numbered steps with clinical parameters: e.g., mm of IPR, 22-hours/day wear, staging criteria).\n"
        "• **Decision rules** (if/then) and **risks/contra-indications**.\n"
        "• **Next steps** (2–4 bullets) and optionally a **chart/task note** (1–2 lines) for the team.\n\n"
        "SAFETY & ETHICS\n"
        "• No patient-specific diagnosis/prescription without adequate clinical information; acknowledge uncertainty briefly. "
        "• Do not address patients. Never invent names, case numbers or internal details."
    )

# =========================
# Minimal RAG Core (safe fallbacks)
# =========================
async def download_db():
    logger.info("download_db(): no-op (override if you pull from cloud)")

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
        logger.info("FAISS not available or index file missing; continuing without FAISS.")

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

# ------------ Translation helpers ------------
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
        logger.warning("OPENAI_API_KEY missing for translation; returning original text.")
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
        if score >= 2:  # strictere end >0
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

# ------------ OpenAI call (async wrapper) ------------
async def get_rag_answer(final_prompt: str) -> str:
    if not OPENAI_API_KEY and not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY is not set; returning fallback answer.")
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
        content = resp.choices[0].message.content or ""
        return content.strip()
    except Exception as e_modern:
        logger.info(f"Modern OpenAI client not used ({e_modern}); trying legacy SDK...")
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
    # 1) Parse body
    try:
        body = await request.json()
    except Exception as e:
        logger.exception(f"Invalid JSON body: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    try:
        logger.info(f"Incoming body keys: {list(body.keys())}")
    except Exception:
        pass

    # 2) Extract user text
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

    # 3) Language + Role
    lang = detect_language(user_text)
    role = detect_role(user_text, lang)
    role_disp = role_label(lang, role)

    # 4) System prompt
    system_prompt = make_system_prompt(lang, role)

    # 5) Cross-lingual retrieval
    retrieval_query = await translate_to_english_if_needed(user_text, lang)
    if retrieval_query != user_text:
        logger.info("Query translated to English for retrieval.")
        logger.info(f"retrieval_query(sample 200): {retrieval_query[:200]}")

    try:
        top_chunks = await get_top_chunks(retrieval_query)
    except Exception as e:
        logger.exception(f"get_top_chunks failed: {e}")
        top_chunks = []

    # FAILSAFE
    if not top_chunks:
        safe_generic = {
            "da": (
                "Her er en enkel, rolletilpasset protokol, når konteksten ikke er tilgængelig.\n\n"
                "1) Screening & diagnose → 2) Planlægning (staging, IPR, attachments) → 3) Start & instruktion "
                "→ 4) Kontroller & tracking → 5) Refinement → 6) Retention.\n\n"
                "Næste skridt: indlæs kliniske detaljer og kilder i RAG for et mere målrettet svar."
            ),
            "de": (
                "Rolleangepasstes Grundprotokoll bei fehlendem Kontext:\n"
                "1) Screening/Diagnose → 2) Planung (Staging, IPR, Attachments) → 3) Start/Instruktion "
                "→ 4) Kontrollen/Tracking → 5) Refinement → 6) Retention.\n\n"
                "Nächste Schritte: klinische Details & Quellen in RAG laden."
            ),
            "fr": (
                "Protocole de base adapté au rôle en l’absence de contexte :\n"
                "1) Dépistage/diagnostic → 2) Planification (staging, IPR, attachments) → 3) Démarrage/instructions "
                "→ 4) Contrôles/tracking → 5) Refinement → 6) Rétention.\n\n"
                "Étapes suivantes : charger les détails cliniques et sources dans le RAG."
            ),
            "en": (
                "Role-adapted baseline protocol when context is missing:\n"
                "1) Screening/diagnosis → 2) Planning (staging, IPR, attachments) → 3) Start/instructions "
                "→ 4) Reviews/tracking → 5) Refinement → 6) Retention.\n\n"
                "Next steps: load clinical details & sources into RAG for a targeted answer."
            ),
        }
        sources = []
        return {
            "finalAnswer": safe_generic.get(lang, safe_generic["en"]),
            "language": lang,
            "role": role_disp,
            "sources": sources,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

    context = "\n\n".join(
        ch.get("text", "") if isinstance(ch, dict) else str(ch)
        for ch in top_chunks
    )[:8000]

    # 6) Final prompt
    final_prompt = (
        f"{system_prompt}\n\n"
        f"IMPORTANT: Use only the information from 'Relevant context' below. "
        f"Do not invent names, case numbers or internal details that are not present in the context.\n\n"
        f"User message:\n{user_text}\n\n"
        f"Relevant context (may be in English and may be partial):\n{context}\n\n"
        f"Answer in the user's language (detected: {lang}, role: {role_disp}):"
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

    # 8) Sources
    sources = []
    for ch in top_chunks[:3]:
        meta = ch.get("meta", {})
        label = _label_from_meta(meta)
        url = meta.get("url") if isinstance(meta, dict) else None
        sources.append({"label": label, "url": url})

    return {
        "finalAnswer": answer,
        "language": lang,
        "role": role_disp,
        "sources": sources,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
