# app/routers/admin_sync.py

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import hmac
import os
import time
import logging

# VIGTIGT: ingen "from app import logger" her (ellers circular import)
logger = logging.getLogger("rag-app")

# Denne funktion må gerne importeres – den trækker ikke app ind
from app.sync.dropbox_sync import sync_once

router = APIRouter(prefix="/admin/sync", tags=["admin-sync"])
bearer = HTTPBearer(auto_error=True)


def _auth(creds: HTTPAuthorizationCredentials = Depends(bearer)) -> bool:
    """
    Simpel bearer-check mod RAG_BEARER_TOKEN env var.
    """
    exp = (os.getenv("RAG_BEARER_TOKEN", "") or "").strip()
    inc = (creds.credentials or "").strip()
    if not exp or not hmac.compare_digest(exp, inc):
        raise HTTPException(status_code=403, detail="Unauthorized")
    return True


@router.get("/ping")
def ping(_: bool = Depends(_auth)):
    return {"ok": True}


@router.get("/inspect")
def inspect(_: bool = Depends(_auth)):
    """
    Læs kun metadata om DB-filen i Dropbox (ingen download).
    Bruger enten ACCESS TOKEN eller Refresh Token flow.
    """
    try:
        import dropbox  # lazy import så modulet ikke kræves ved runtime uden /inspect
        dbx = None

        tok = (os.getenv("DROPBOX_ACCESS_TOKEN") or "").strip()
        rf = (os.getenv("DROPBOX_REFRESH_TOKEN") or "").strip()
        cid = (os.getenv("DROPBOX_CLIENT_ID") or "").strip()
        sec = (os.getenv("DROPBOX_CLIENT_SECRET") or "").strip()

        if tok:
            dbx = dropbox.Dropbox(tok)
        elif rf and cid and sec:
            dbx = dropbox.Dropbox(
                oauth2_refresh_token=rf,
                app_key=cid,
                app_secret=sec,
            )
        else:
            raise RuntimeError("Dropbox credentials missing")

        path = (os.getenv("DROPBOX_DB_PATH") or "").strip()
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


def _sync_job():
    """
    Baggrundsjob for sync. Kører sync_once og logger resultat.
    """
    logger.info("SYNC: background job started")
    try:
        res = sync_once(logger)
        logger.info(f"SYNC: background job result={res}")
    except Exception:
        logger.exception("SYNC: background job crashed")


@router.post("/run")
def run_sync(
    background: BackgroundTasks,
    _: bool = Depends(_auth),
    mode: str = "bg",
):
    """
    mode=bg (default): starter baggrunds-job og returnerer straks {accepted:true}
    mode=fg: kør synkront og returnér resultatet (kan time out ved store filer)
    """
    if mode == "fg":
        logger.info("SYNC: /admin/sync/run (foreground)")
        try:
            res = sync_once(logger)
            return res
        except Exception as e:
            logger.exception("SYNC: foreground failed")
            raise HTTPException(status_code=500, detail=str(e))
    else:
        logger.info("SYNC: /admin/sync/run (background accepted)")
        background.add_task(_sync_job)
        return {"accepted": True, "ts": int(time.time())}
