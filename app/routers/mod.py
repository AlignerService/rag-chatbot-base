# app/routers/mod.py
from fastapi import APIRouter, HTTPException, Request, Depends
import aiosqlite, os
from datetime import datetime
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import hmac

router = APIRouter(prefix="/mod", tags=["moderation"])

RAG_BEARER_TOKEN = os.getenv("RAG_BEARER_TOKEN","")
bearer = HTTPBearer(auto_error=True)

def _auth(creds: HTTPAuthorizationCredentials):
    if not RAG_BEARER_TOKEN or creds.scheme.lower() != "bearer":
        raise HTTPException(status_code=403, detail="No token")
    if not hmac.compare_digest(creds.credentials.strip(), RAG_BEARER_TOKEN.strip()):
        raise HTTPException(status_code=403, detail="Bad token")

DB_PATH = os.getenv("DB_PATH", "/data/rag.sqlite3")

async def _ensure_schema(db):
    # 1) Opret tabel hvis den ikke findes
    await db.execute("""
        CREATE TABLE IF NOT EXISTS moderation_queue(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          created_at TEXT NOT NULL,
          status TEXT NOT NULL,
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
          -- bemærk: ingen final_public her; den tilføjes via migration nedenfor
        )
    """)
    await db.execute("CREATE INDEX IF NOT EXISTS idx_mq_status_id ON moderation_queue(status, id)")

    # 2) Migration: tilføj manglende kolonner dynamisk
    async def has_col(name: str) -> bool:
        async with db.execute("PRAGMA table_info('moderation_queue')") as cur:
            async for r in cur:
                if (r[1] or "").lower() == name.lower():
                    return True
        return False

    # final_public til offentlig visning
    if not await has_col("final_public"):
        await db.execute("ALTER TABLE moderation_queue ADD COLUMN final_public TEXT")
    # language hvis vi senere vil gemme sprog (ufarligt at have den med)
    if not await has_col("language"):
        await db.execute("ALTER TABLE moderation_queue ADD COLUMN language TEXT")

    # ingen commit her; kalderen committer efter opdatering/insert

@router.post("/intake")
async def mod_intake(request: Request, credentials: HTTPAuthorizationCredentials = Depends(bearer)):
    _auth(credentials)
    body = await request.json()
    when = datetime.utcnow().isoformat() + "Z"

    session_id   = (body.get("sessionId") or "").strip()
    ticket_id    = (body.get("ticketId") or "").strip()
    contact_id   = (body.get("contactId") or "").strip()
    from_email   = (body.get("fromEmail") or "").strip().lower()
    contact_name = (body.get("contactName") or body.get("customerName") or "").strip()
    subject      = (body.get("subject") or "").strip()
    user_text    = (body.get("question") or body.get("text") or "").strip()
    language     = (body.get("lang") or "").strip().lower()

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
    return {"ok": True, "status": status, "id": mid}

@router.get("/queue")
async def mod_queue(status: str = "pending", limit: int = 50, credentials: HTTPAuthorizationCredentials = Depends(bearer)):
    _auth(credentials)
    q = "SELECT id,created_at,status,from_email,contact_name,subject,user_text,model_answer,editor_answer FROM moderation_queue "
    args = []
    if status and status.lower() != "any":
        q += "WHERE status=? "
        args.append(status.lower())
    q += "ORDER BY id DESC LIMIT ?"
    args.append(max(1, min(limit, 200)))
    rows=[]
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        await _ensure_schema(db)
        async with db.execute(q, tuple(args)) as cur:
            rows = [dict(r) for r in await cur.fetchall()]
    return {"ok": True, "rows": rows}

@router.get("/item/{mid}")
async def mod_item(mid: int, credentials: HTTPAuthorizationCredentials = Depends(bearer)):
    _auth(credentials)
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        await _ensure_schema(db)
        async with db.execute("SELECT * FROM moderation_queue WHERE id=?", (mid,)) as cur:
            row = await cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="not found")
    return {"ok": True, "item": dict(row)}

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
        # 1) find status + tekst
        async with db.execute("SELECT status, editor_answer FROM moderation_queue WHERE id=?", (mid,)) as cur:
            row = await cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="not found")
        if (row["status"] or "").lower() == "approved":
            raise HTTPException(status_code=409, detail="already approved")
        final_text = (row["editor_answer"] or "").strip()

        # 2) update (status + final_public spejl)
        await db.execute(
            "UPDATE moderation_queue SET status='approved', approved_by=?, approved_at=?, final_public=? WHERE id=?",
            (who, when, final_text, mid)
        )
        await db.commit()
    return {"ok": True, "final": final_text, "id": mid}

# Offentlig, uden token: bruges af kundesiden til at hente godkendt svar
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
