import os, time, tempfile
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
    import time as _time
    path = path if path.startswith("/") else f"/{path}"
    md = dbx.files_get_metadata(path)
    if hasattr(md, "server_modified"):
        return int(md.server_modified.timestamp()), getattr(md, "rev", None)
    return int(_time.time()), getattr(md, "rev", None)

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

# === NY VERSION: atomisk swap på SAMME filesystem ===
def _atomic_swap(tmp_file: Path, final_file: Path):
    # Kør KUN når tmp_file og final_file er på samme FS
    final_file.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp_file, final_file)  # atomisk på samme filesystem

def sync_once(logger) -> dict:
    """Pull fra Dropbox til /data, hvis der er ny fil. Returnerer status-dict."""
    from dropbox.exceptions import ApiError

    db_local = Path(os.getenv("LOCAL_DB_PATH","/data/rag.sqlite3"))
    db_drop = os.getenv("DROPBOX_DB_PATH","/rag.sqlite3").strip()
    min_age = int(os.getenv("SYNC_MIN_AGE_SEC","120"))

    # 1) Auth
    try:
        dbx = _get_dbx()
    except Exception as e:
        logger.exception("Dropbox auth failed")
        return {"ok": False, "error": f"auth: {e}"}

    # 2) Metadata
    try:
        remote_mtime, remote_rev = _modified_epoch_rev(dbx, db_drop)
    except ApiError as e:
        logger.exception("Metadata read failed")
        return {"ok": False, "error": f"metadata: {e}"}

    now = int(time.time())
    if now - remote_mtime < min_age:
        return {"ok": True, "skipped": True, "reason": "remote-too-fresh"}

    # 3) Rev-markør: undgå unødige downloads
    marker = db_local.with_suffix(".rev.txt")
    local_rev = marker.read_text().strip() if marker.exists() else ""
    if local_rev == (remote_rev or "") and db_local.exists():
        return {"ok": True, "skipped": True, "reason": "same-rev"}

    # 4) Hent direkte til tempfil i SAMME mappe som destinationen (/data)
    final_dir = db_local.parent
    final_dir.mkdir(parents=True, exist_ok=True)

    fd, tmp_path_str = tempfile.mkstemp(prefix="db.", suffix=".tmp", dir=str(final_dir))
    os.close(fd)
    tmp_file = Path(tmp_path_str)

    try:
        # hent direkte til tmp-filen i /data
        dbx_path = db_drop if db_drop.startswith("/") else f"/{db_drop}"
        dbx.files_download_to_file(str(tmp_file), dbx_path)

        # 5) Integritetscheck
        if not _sqlite_integrity_ok(tmp_file):
            try:
                tmp_file.unlink()
            except Exception:
                pass
            return {"ok": False, "error": "integrity_check_failed"}

        # 6) Atomisk swap
        _atomic_swap(tmp_file, db_local)

        # 6b) Stramme filrettigheder (valgfrit, men rart)
        try:
            os.chmod(db_local, 0o600)
        except Exception:
            pass

        # 7) Skriv rev-markør
        try:
            marker.write_text(remote_rev or "")
        except Exception:
            pass

        return {"ok": True, "updated": True, "path": str(db_local), "rev": remote_rev}

    except Exception as e:
        # oprydning på fejl
        try:
            if tmp_file.exists():
                tmp_file.unlink()
        except Exception:
            pass
        return {"ok": False, "error": str(e)}
