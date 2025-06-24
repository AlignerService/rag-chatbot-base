import aiosqlite
from typing import List
from datetime import datetime

LOCAL_DB_PATH = None

def init_db_path(path: str):
    global LOCAL_DB_PATH
    LOCAL_DB_PATH = path

async def get_ticket_history(ticket_id: str) -> List[str]:
    """
    Henter tidligere samtaler for et enkelt ticket_id i kronologisk rækkefølge.
    Returnerer en liste af tekststykker.
    """
    if LOCAL_DB_PATH is None:
        raise RuntimeError("Database path ikke initialiseret. Kald init_db_path() først.")
    rows = []
    async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
        query = '''
            SELECT sender || ': ' || content
            FROM ticket_threads
            WHERE ticket_id = ?
            ORDER BY datetime(created_time) ASC
        '''
        async with conn.execute(query, (ticket_id,)) as cur:
            async for row in cur:
                rows.append(row[0])
    return rows

async def get_customer_history(contact_id: str) -> List[str]:
    """
    Henter tidligere samtaler på tværs af alle ticket_id for en kontakt,
    ekskluderer eventuelt den aktuelle ticket, hvis ønsket.
    Returnerer en liste af tekststykker.
    """
    if LOCAL_DB_PATH is None:
        raise RuntimeError("Database path ikke initialiseret. Kald init_db_path() først.")
    rows = []
    async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
        query = '''
            SELECT sender || ': ' || content
            FROM ticket_threads
            WHERE contact_id = ?
            ORDER BY datetime(created_time) ASC
        '''
        async with conn.execute(query, (contact_id,)) as cur:
            async for row in cur:
                rows.append(row[0])
    return rows
