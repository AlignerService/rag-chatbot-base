# app/api_search_helpers.py

import aiosqlite
from typing import List, Optional

# Will be initialized from app.py on startup
LOCAL_DB_PATH: Optional[str] = None

def init_db_path(path: str):
    """
    Initialize the module-level database path. Must be called before using
    get_ticket_history or get_customer_history.
    """
    global LOCAL_DB_PATH
    LOCAL_DB_PATH = path

async def get_ticket_history(ticket_id: str) -> List[str]:
    """
    Fetches all message threads for a single ticket in chronological order.
    Returns a list of strings like "sender: content".
    """
    if LOCAL_DB_PATH is None:
        raise RuntimeError("Database path not initialized. Call init_db_path() first.")
    rows: List[str] = []
    query = """
SELECT
  sender || ': ' || content
FROM ticket_threads
WHERE ticket_id = ?
ORDER BY datetime(created_time) ASC
"""
    async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
        async with conn.execute(query, (ticket_id,)) as cur:
            async for row in cur:
                rows.append(row[0])
    return rows

async def get_customer_history(
    contact_id: str,
    exclude_ticket_id: Optional[str] = None
) -> List[str]:
    """
    Fetches all message threads for a given contact across tickets,
    optionally excluding the current ticket. Returns a list of strings
    like "sender: content" in chronological order.
    """
    if LOCAL_DB_PATH is None:
        raise RuntimeError("Database path not initialized. Call init_db_path() first.")
    rows: List[str] = []

    if exclude_ticket_id:
        query = """
SELECT
  sender || ': ' || content
FROM ticket_threads
WHERE contact_id = ? AND ticket_id != ?
ORDER BY datetime(created_time) ASC
"""
        params = (contact_id, exclude_ticket_id)
    else:
        query = """
SELECT
  sender || ': ' || content
FROM ticket_threads
WHERE contact_id = ?
ORDER BY datetime(created_time) ASC
"""
        params = (contact_id,)

    async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
        async with conn.execute(query, params) as cur:
            async for row in cur:
                rows.append(row[0])
    return rows
