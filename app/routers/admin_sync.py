from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import hmac, os, time
from app.sync.dropbox_sync import sync_once
from app import logger  # din logger i app/__init__.py

router = APIRouter(prefix="/admin/sync", tags=["admin-sync"])
bearer = HTTPBearer(auto_error=True)

def _auth(creds: HTTPAuthorizationCredentials = Depends(bearer)):
    exp = (os.getenv("RAG_BEARER_TOKEN","") or "").strip()
    inc = (creds.credentials or "").strip()
    if not exp or not hmac.compare_digest(exp, inc):
        raise HTTPException(status_code=403, detail="Unauthorized")
    return True

@router.get("/ping")
def ping(_: bool = Depends(_auth)):
    return {"ok": True}

# ——— INSPECT: læs kun metadata fra Dropbox (ingen download) ———
@router.get("/inspect")
def inspect(_: bool = Depends(_auth)):
    try:
        import dropbox
        dbx = None
        tok = os.getenv("DROPBOX_ACCESS_TOKEN","").strip()
        if tok:
            dbx = dropbox.Dropbox(tok)
        else:
            rf = os.getenv("DROPBOX_REFRESH_TOKEN","").strip()
            cid = os.getenv("DROPBOX_CLIENT_ID","").strip()
            sec = os.getenv("DROPBOX_CLIENT_SECRET","").strip()
            if rf and cid and sec:
                dbx = dropbox.Dropbox(oauth2_refresh_token=rf, app_key=cid, app_secret=sec)
        if not dbx:
            raise RuntimeError("Dropbox credentials missing")

        path = os.getenv("DROPBOX_DB_PATH","").strip()
        if not path:
            raise RuntimeError("DROPBOX_DB_PATH missing")

        if not path.startswith("/"):
            path = f"/{path}"

        md = dbx.files_get_metadata(path)
        size = getattr(md, "size", None)
        rev = getattr(md, "rev", None)
        server_modified = getattr(md, "server_modified", None)
        return {
            "ok": True,
            "dropbox_path": path,
            "size_bytes": size,
            "rev": rev,
            "server_modified": server_modified.isoformat() if server_modified else None,
        }
    except Exception as e:
        logger.exception("INSPECT failed")
        raise HTTPException(status_code=500, detail=str(e))

# ——— RUN: kør sync i baggrunden (returnér straks) ———
def _sync_job():
    logger.info("SYNC: background job started")
    res = sync_once(logger)
    logger.info(f"SYNC: background job result={res}")

@router.post("/run")
def run_sync(background: BackgroundTasks, _: bool = Depends(_auth), mode: str = "bg"):
    """
    mode=bg (default): starter baggrunds-job og returnerer straks {accepted:true}
    mode=fg: kør synkron og returnér resultatet (kan time out, ikke anbefalet til store filer)
    """
    if mode == "fg":
        logger.info("SYNC: /admin/sync/run (foreground)")
        res = sync_once(logger)
        return res
    else:
        logger.info("SYNC: /admin/sync/run (background accepted)")
        background.add_task(_sync_job)
        return {"accepted": True, "ts": int(time.time())}
