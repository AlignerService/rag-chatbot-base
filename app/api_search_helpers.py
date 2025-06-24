import aiosqlite
from typing import List
from datetime import datetime

LOCAL_DB_PATH = None

def init_db_path(path: str):
    """
    Initialize the module-level database path. Call this once on startup.
    """
    global LOCAL_DB_PATH
    LOCAL_DB_PATH = path

async def get_ticket_history(ticket_id: str) -> List[str]:
    """
    Retrieve chronological history (sender: content) for a specific ticket_id.
    """
    if LOCAL_DB_PATH is None:
        raise RuntimeError("Database path not initialized. Call init_db_path() first.")
    rows: List[str] = []
    query = '''
        SELECT sender || ': ' || content
        FROM ticket_threads
        WHERE ticket_id = ?
        ORDER BY datetime(created_time) ASC
    '''
    async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
        async with conn.execute(query, (ticket_id,)) as cursor:
            async for record in cursor:
                rows.append(record[0])
    return rows

async def get_customer_history(contact_id: str, exclude_ticket_id: str = None) -> List[str]:
    """
    Retrieve chronological history (sender: content) across all tickets for a contact_id,
    optionally excluding the current ticket.
    """
    if LOCAL_DB_PATH is None:
        raise RuntimeError("Database path not initialized. Call init_db_path() first.")
    rows: List[str] = []
    # Build query
    base = '''
        SELECT sender || ': ' || content
        FROM ticket_threads
        WHERE contact_id = ?
    '''
    if exclude_ticket_id:
        base += " AND ticket_id != ?"
        params = (contact_id, exclude_ticket_id)
    else:
        params = (contact_id,)
    base += " ORDER BY datetime(created_time) ASC"

    async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
        async with conn.execute(base, params) as cursor:
            async for record in cursor:
                rows.append(record[0])
    return rows
