# app/routers/mod.py (kun relevante bider)
from fastapi import APIRouter, HTTPException, Request, Depends
import aiosqlite, os, json
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

DB_PATH = os.getenv("LOCAL_DB_PATH", "/data/rag.sqlite3")

@router.post("/intake")
async def mod_intake(request: Request, credentials: HTTPAuthorizationCredentials = Depends(bearer)):
    _auth(credentials)
    body = await request.json()
    when = datetime.utcnow().isoformat() + "Z"
    # Standardiser felter
    session_id  = (body.get("sessionId") or "").strip()
    ticket_id   = (body.get("ticketId") or "").strip()
    contact_id  = (body.get("contactId") or "").strip()
    from_email  = (body.get("fromEmail") or "").strip().lower()
    contact_name= (body.get("contactName") or body.get("customerName") or "").strip()
    subject     = (body.get("subject") or "").strip()
    user_text   = (body.get("question") or body.get("text") or "").strip()

    if not user_text:
        raise HTTPException(status_code=400, detail="missing user text")

    # status SKAL være 'pending'
    status = "pending"

    async with aiosqlite.connect(DB_PATH) as db:
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
            )
        """)
        await db.execute("CREATE INDEX IF NOT EXISTS idx_mq_status_id ON moderation_queue(status, id)")

        await db.execute(
            """INSERT INTO moderation_queue
               (created_at,status,session_id,ticket_id,contact_id,from_email,contact_name,subject,user_text,model_answer,editor_answer)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (when,status,session_id,ticket_id,contact_id,from_email,contact_name,subject,user_text,"","")
        )
        await db.commit()
    return {"ok": True, "status": status}

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
        async with db.execute(q, tuple(args)) as cur:
            rows = [dict(r) for r in await cur.fetchall()]
    return {"ok": True, "rows": rows}

@router.post("/save")
async def mod_save(payload: dict, credentials: HTTPAuthorizationCredentials = Depends(bearer)):
    _auth(credentials)
    mid = int(payload.get("id") or 0)
    txt = (payload.get("editor_answer") or "").strip()
    if not mid:
        raise HTTPException(status_code=400, detail="missing id")
    async with aiosqlite.connect(DB_PATH) as db:
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
        # 1) find status
        async with db.execute("SELECT status FROM moderation_queue WHERE id=?", (mid,)) as cur:
            row = await cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="not found")
        if (row["status"] or "").lower() == "approved":
            raise HTTPException(status_code=409, detail="already approved")
        # 2) update
        await db.execute(
            "UPDATE moderation_queue SET status='approved', approved_by=?, approved_at=? WHERE id=?",
            (who, when, mid)
        )
        # 3) hent final tekst til frontend UX
        async with db.execute("SELECT editor_answer FROM moderation_queue WHERE id=?", (mid,)) as cur:
            row2 = await cur.fetchone()
        await db.commit()
    final = (row2["editor_answer"] if row2 else "") or ""
    return {"ok": True, "final": final}

