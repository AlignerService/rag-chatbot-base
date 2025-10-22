import os, time, tempfile, shutil
from pathlib import Path

def _get_dbx():
    import dropbox
    tok = os.getenv("DROPBOX_ACCESS_TOKEN","").strip()
    if tok:
        return dropbox.Dropbox(tok)
    rf = os.getenv("DROPBOX_REFRESH_TOKEN","").strip()
    cid = os.getenv("DROPBOX_CLIENT_ID","").strip()
    sec = os.getenv("DROPBOX_CLIENT_SECRET","").strip()
    if rf and cid and sec:
        return dropbox.Dropbox(oauth2_refresh_token=rf, app_key=cid, app_secret=sec)
    raise RuntimeError("Dropbox credentials missing")

def _modified_epoch_rev(dbx, path: str):
    path = path if path.startswith("/") else f"/{path}"
    md = dbx.files_get_metadata(path)
    if hasattr(md, "server_modified"):
        return int(md.server_modified.timestamp()), getattr(md, "rev", None)
    return int(time.time()), getattr(md, "rev", None)

def _sqlite_integrity_ok(p: Path) -> bool:
    import sqlite3
    try:
        con = sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=5)
        try:
            cur = con.execute("PRAGMA integrity_check;")
            row = cur.fetchone()
            return bool(row and row[0] == "ok")
        finally:
            con.close()
    except Exception:
        return False

def _atomic_swap(tmp_file: Path, final_file: Path):
    final_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file.replace(final_file)  # atomisk på samme filesystem

def sync_once(logger) -> dict:
    """Pull fra Dropbox til /data, hvis der er ny fil. Returnerer status-dict."""
    from dropbox.exceptions import ApiError
    db_local = Path(os.getenv("LOCAL_DB_PATH","/data/rag.sqlite3"))
    db_drop = os.getenv("DROPBOX_DB_PATH","/rag.sqlite3").strip()
    min_age = int(os.getenv("SYNC_MIN_AGE_SEC","120"))

    try:
        dbx = _get_dbx()
    except Exception as e:
        logger.exception("Dropbox auth failed")
        return {"ok": False, "error": f"auth: {e}"}

    try:
        remote_mtime, remote_rev = _modified_epoch_rev(dbx, db_drop)
    except ApiError as e:
        logger.exception("Metadata read failed")
        return {"ok": False, "error": f"metadata: {e}"}

    now = int(time.time())
    if now - remote_mtime < min_age:
        return {"ok": True, "skipped": True, "reason": "remote-too-fresh"}

    # rev-markør for at undgå unødige downloads
    marker = db_local.with_suffix(".rev.txt")
    local_rev = marker.read_text().strip() if marker.exists() else ""
    if local_rev == (remote_rev or "") and db_local.exists():
        return {"ok": True, "skipped": True, "reason": "same-rev"}

    # hent til tmp
    tmp_dir = Path(tempfile.mkdtemp())
    tmp_file = tmp_dir / "db.tmp"
    try:
        with open(tmp_file, "wb") as f:
            dbx.files_download_to_file(f.name, db_drop if db_drop.startswith("/") else f"/{db_drop}")

        if not _sqlite_integrity_ok(tmp_file):
            return {"ok": False, "error": "integrity_check_failed"}

        _atomic_swap(tmp_file, db_local)
        try:
            marker.write_text(remote_rev or "")
        except Exception:
            pass

        return {"ok": True, "updated": True, "path": str(db_local), "rev": remote_rev}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    finally:
        try:
            if tmp_file.exists(): tmp_file.unlink(missing_ok=True)
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass
