from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import hmac, os
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

@router.post("/run")
def run_sync(_: bool = Depends(_auth)):
    return sync_once(logger)
