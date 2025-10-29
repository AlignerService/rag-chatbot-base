# app/routers/mod.py
from fastapi import APIRouter, HTTPException, Request, Depends
import aiosqlite, os, hmac, re, logging, uuid
from datetime import datetime
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import asyncio, aiohttp

router = APIRouter(prefix="/mod", tags=["moderation"])

# ------------------
# Config & logging
# ------------------
logger = logging.getLogger("rag-mod")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

RAG_BEARER_TOKEN = os.getenv("RAG_BEARER_TOKEN", "").strip()
bearer = HTTPBearer(auto_error=True)

def _auth(creds: HTTPAuthorizationCredentials):
    if not RAG_BEARER_TOKEN or creds.scheme.lower() != "bearer":
        raise HTTPException(status_code=403, detail="No token")
    if not hmac.compare_digest(creds.credentials.strip(), RAG_BEARER_TOKEN):
        raise HTTPException(status_code=403, detail="Bad token")

DB_PATH = os.getenv("DB_PATH", "/data/rag.sqlite3").strip()

# ------------------
# Autosuggest setup
# ------------------
SELF_BASE = os.getenv("SELF_BASE_URL", "https://alignerservice-rag.onrender.com").rstrip("/")
SUGGEST_TIMEOUT = int(os.getenv("SUGGEST_TIMEOUT_SEC", "20") or "20")

async def _autosuggest(mid: int, question: str, lang: str):
    """Call own /api/answer and store into moderation_queue.model_answer. Fire-and-forget."""
    if not question:
        return
    try:
        payload = {"question": question, "output_mode": "mail", "lang": (lang or "").strip() or "da"}
        headers = {"Content-Type": "application/json"}
        if RAG_BEARER_TOKEN:
            headers["Authorization"] = f"Bearer {RAG_BEARER_TOKEN}"
        timeout = aiohttp.ClientTimeout(total=SUGGEST_TIMEOUT)

        txt = ""
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{SELF_BASE}/api/answer", json=payload, headers=headers) as resp:
                if resp.status == 200:
                    try:
                        data = await resp.json()
                    except Exception:
                        data = {}
                    txt = (data.get("finalAnswerPlain")
                           or data.get("finalAnswerMarkdown")
                           or data.get("finalAnswer")
                           or data.get("text")
                           or "").strip()
                if not txt:
                    logger.debug(f"_autosuggest(mid={mid}) empty response, status={resp.status}")

        if txt:
            async with aiosqlite.connect(DB_PATH) as db:
                await _ensure_schema(db)
                await db.execute("UPDATE moderation_queue SET model_answer=? WHERE id=?", (txt, mid))
                await db.commit()
                logger.debug(f"_autosuggest(mid={mid}) wrote {len(txt)} chars")
    except Exception as e:
        logger.debug(f"_autosuggest(mid={mid}) failed: {e}")

# ------------------
# DB schema helpers
# ------------------
async def _ensure_schema(db: aiosqlite.Connection):
    # stable SQLite settings
    try:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA synchronous=NORMAL")
        await db.execute("PRAGMA busy_timeout=5000")
    except Exception:
        pass

    # migrate under write-lock
    try:
        await db.execute("BEGIN IMMEDIATE")
    except Exception:
        pass

    await db.execute("""
        CREATE TABLE IF NOT EXISTS moderation_queue(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          created_at TEXT NOT NULL,
          status TEXT NOT NULL,   -- pending | approved
          session_id TEXT,
          ticket_id TEXT,
          contact_id TEXT,
          from_email TEXT,
          contact_name TEXT,
          subject TEXT,
          user_text TEXT,
          model_answer TEXT,
          editor_answer TEXT,
          approved_by TEXT,
          approved_at TEXT
        )
    """)
    await db.execute("CREATE INDEX IF NOT EXISTS idx_mq_status_id ON moderation_queue(status, id)")
    # NYT: hurtig historik-opslag pr. session
    await db.execute("CREATE INDEX IF NOT EXISTS idx_mq_session_id ON moderation_queue(session_id)")

    # ensure optional columns
    cols = set()
    async with db.execute("PRAGMA table_info('moderation_queue')") as cur:
        async for r in cur:
            cols.add((r[1] or "").lower())

    async def add_col(name: str, ddl: str):
        if name.lower() in cols:
            return
        try:
            await db.execute(f"ALTER TABLE moderation_queue ADD COLUMN {ddl}")
            cols.add(name.lower())
        except Exception:
            again = set()
            async with db.execute("PRAGMA table_info('moderation_queue')") as cur2:
                async for r2 in cur2:
                    again.add((r2[1] or "").lower())
            if name.lower() not in again:
                raise

    await add_col("final_public", "final_public TEXT")
    await add_col("language", "language TEXT")

    try:
        await db.execute("CREATE INDEX IF NOT EXISTS idx_mq_lang ON moderation_queue(language)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_mq_created ON moderation_queue(created_at)")
    except Exception:
        pass

    try:
        await db.commit()
    except Exception:
        pass

# ------------------
# Helpers: greeting name
# ------------------
def _name_from_email(email: str) -> str:
    """
    Derive a human-looking name from email local-part.
    john.doe_smith -> 'John Doe Smith'
    """
    local = (email or "").split("@")[0]
    local = re.sub(r"[._\\-]+", " ", local).strip()
    # collapse multiple spaces
    local = re.sub(r"\\s{2,}", " ", local)
    return local.title() if local else ""

def greeting_for(lang: str, name: str, email: str) -> str:
    nm = (name or "").strip() or _name_from_email(email or "")
    if lang == "da":
        return f"Hej {nm}," if nm else "Hej,"
    if lang == "de":
        return f"Hallo {nm}," if nm else "Hallo,"
    if lang == "fr":
        return f"Bonjour {nm}," if nm else "Bonjour,"
    # default EN
    return f"Hi {nm}," if nm else "Hi,"

# ------------------
# Endpoints
# ------------------
@router.post("/intake")
async def mod_intake(request: Request, credentials: HTTPAuthorizationCredentials = Depends(bearer)):
    _auth(credentials)
    body = await request.json()
    when = datetime.utcnow().isoformat() + "Z"

    session_id   = (body.get("sessionId") or "").strip()
    if not session_id:
        session_id = str(uuid.uuid4())

    ticket_id    = (body.get("ticketId") or "").strip()
    contact_id   = (body.get("contactId") or "").strip()
    from_email   = (body.get("fromEmail") or "").strip().lower()
    contact_name = (body.get("contactName") or body.get("customerName") or "").strip()
    subject      = (body.get("subject") or "").strip()
    user_text    = (body.get("question") or body.get("text") or "").strip()
    language     = (body.get("language") or body.get("lang") or "").strip().lower()

    if not user_text:
        raise HTTPException(status_code=400, detail="missing user text")

    status = "pending"

    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        cur = await db.execute(
            """INSERT INTO moderation_queue
               (created_at,status,session_id,ticket_id,contact_id,from_email,contact_name,subject,user_text,model_answer,editor_answer,final_public,language)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (when,status,session_id,ticket_id,contact_id,from_email,contact_name,subject,user_text,"","", "", language)
        )
        await db.commit()
        mid = cur.lastrowid or 0

    # fire-and-forget autosuggest (non-blocking)
    try:
        asyncio.create_task(_autosuggest(mid, user_text, language))
    except Exception:
        pass

    return {"ok": True, "status": status, "id": mid, "sessionId": session_id}


@router.get("/queue")
async def mod_queue(status: str = "pending", limit: int = 50, credentials: HTTPAuthorizationCredentials = Depends(bearer)):
    _auth(credentials)
    # Inkluder session_id i SELECT (kritisk for Moderator-historik)
    q = ("SELECT id,created_at,status,session_id,from_email,contact_name,subject,"
         "user_text,model_answer,editor_answer FROM moderation_queue ")
    args = []
    if status and status.lower() != "any":
        q += "WHERE status=? "
        args.append(status.lower())
    q += "ORDER BY id DESC LIMIT ?"
    args.append(max(1, min(int(limit or 50), 200)))
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        await _ensure_schema(db)
        async with db.execute(q, tuple(args)) as cur:
            rows = [dict(r) for r in await cur.fetchall()]
    return {"ok": True, "rows": rows}


@router.get("/suggest/{mid}")
async def mod_suggest(mid: int, credentials: HTTPAuthorizationCredentials = Depends(bearer)):
    _auth(credentials)
    # kick baggrundsjob hvis der ikke findes et forslag endnu
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        await _ensure_schema(db)  # READ, ikke write
        async with db.execute("SELECT user_text, model_answer, language FROM moderation_queue WHERE id=?", (mid,)) as cur:
            row = await cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="not found")

    if not (row["model_answer"] or "").strip():
        try:
            asyncio.create_task(_autosuggest(mid, row["user_text"] or "", (row["language"] or "")))
        except Exception:
            pass

    # kort poll (≤ 4 sek) for at levere noget “nu”
    for _ in range(4):
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            # READ, ikke write
            async with db.execute("SELECT model_answer FROM moderation_queue WHERE id=?", (mid,)) as cur:
                r = await cur.fetchone()
                txt = (r and (r["model_answer"] or "").strip()) or ""
                if txt:
                    return {"ok": True, "suggestion": txt}
        await asyncio.sleep(1)

    # stadig intet? Så svar hurtigt og lad frontenden prøve igen om lidt
    return {"ok": False, "suggestion": ""}


@router.post("/save")
async def mod_save(payload: dict, credentials: HTTPAuthorizationCredentials = Depends(bearer)):
    _auth(credentials)
    mid = int(payload.get("id") or 0)
    txt = (payload.get("editor_answer") or "").strip()
    if not mid:
        raise HTTPException(status_code=400, detail="missing id")
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        await db.execute("UPDATE moderation_queue SET editor_answer=? WHERE id=?", (txt, mid))
        await db.commit()
    return {"ok": True}


@router.post("/approve")
async def mod_approve(payload: dict, credentials: HTTPAuthorizationCredentials = Depends(bearer)):
    _auth(credentials)
    mid = int(payload.get("id") or 0)
    who = (payload.get("approved_by") or "Staff").strip()
    when = datetime.utcnow().isoformat()+"Z"
    if not mid:
        raise HTTPException(status_code=400, detail="missing id")

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        await _ensure_schema(db)

        # include from_email for greeting fallback
        async with db.execute(
            "SELECT status, editor_answer, contact_name, language, from_email FROM moderation_queue WHERE id=?", (mid,)
        ) as cur:
            row = await cur.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="not found")
        if (row["status"] or "").lower() == "approved":
            raise HTTPException(status_code=409, detail="already approved")

        editor_text = (row["editor_answer"] or "").strip()
        lang = (row["language"] or "").strip().lower()
        name = (row["contact_name"] or "").strip()
        email = (row["from_email"] or "").strip()

        # Replace a bare greeting among the first lines, else prepend a proper one
        lines = editor_text.splitlines()
        greet_pat = re.compile(r'^\s*(hi|hej|hallo|bonjour)\s*,?\s*$', re.IGNORECASE)
        replaced = False
        for idx in range(min(len(lines), 6)):
            if greet_pat.match(lines[idx] or ""):
                lines[idx] = greeting_for(lang, name, email)
                replaced = True
                break
        if not replaced:
            starts = (editor_text[:12] or "").lower()
            has_greet = any(starts.startswith(x) for x in ["hej", "hi", "hallo", "bonjour"])
            if has_greet:
                final_text = editor_text
            else:
                final_text = f"{greeting_for(lang, name, email)}\n\n{editor_text}".strip()
        else:
            final_text = "\n".join(lines).strip()

        await db.execute(
            "UPDATE moderation_queue SET status='approved', approved_by=?, approved_at=?, final_public=? WHERE id=?",
            (who, when, final_text, mid)
        )
        await db.commit()

    return {"ok": True, "final": final_text, "id": mid}


@router.get("/public")
async def mod_public(id: int):
    if not id:
        raise HTTPException(status_code=400, detail="missing id")
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        await _ensure_schema(db)
        async with db.execute("SELECT status, final_public FROM moderation_queue WHERE id=?", (id,)) as cur:
            row = await cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="not found")
    return {"ok": True, "status": row["status"], "answer": row["final_public"] or ""}


@router.get("/health")
async def mod_health(credentials: HTTPAuthorizationCredentials = Depends(bearer)):
    _auth(credentials)
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await _ensure_schema(db)
            async with db.execute("SELECT 1") as cur:
                await cur.fetchone()
        return {"ok": True, "db": "ready"}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


@router.get("/history")
async def mod_history(session_id: str, limit: int = 100, credentials: HTTPAuthorizationCredentials = Depends(bearer)):
    _auth(credentials)
    sid = (session_id or "").strip()
    if not sid:
        raise HTTPException(status_code=400, detail="missing session_id")
    lim = max(1, min(int(limit or 100), 500))

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        await _ensure_schema(db)
        sql = (
            "SELECT id, created_at, status, session_id, from_email, contact_name, subject, "
            "user_text, model_answer, editor_answer, final_public, language "
            "FROM moderation_queue WHERE session_id = ? "
            "ORDER BY id ASC LIMIT ?"
        )
        async with db.execute(sql, (sid, lim)) as cur:
            rows = [dict(r) for r in await cur.fetchall()]
    return {"ok": True, "rows": rows}
