import os
import aiosqlite
from typing import List

# Hent database-sti fra miljøvariabler, default til lokal temp-fil
LOCAL_DB_PATH = os.getenv("LOCAL_DB_PATH", "/tmp/knowledge.sqlite")

async def get_ticket_history(ticket_id: str) -> List[str]:
    """
    Henter tidligere samtaler for et enkelt ticket_id i kronologisk rækkefølge.
    Returnerer en liste af tekststykker med afsender.
    """
    rows: List[str] = []
    query = '''
        SELECT sender || ': ' || content
        FROM ticket_threads
        WHERE ticket_id = ?
        ORDER BY datetime(created_time) ASC
    '''
    async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
        async with conn.execute(query, (ticket_id,)) as cursor:
            async for row in cursor:
                rows.append(row[0])
    return rows

async def get_customer_history(contact_id: str) -> List[str]:
    """
    Henter tidligere samtaler på tværs af alle ticket_id for en kontakt.
    Returnerer en liste af tekststykker med afsender.
    """
    rows: List[str] = []
    query = '''
        SELECT sender || ': ' || content
        FROM ticket_threads
        WHERE contact_id = ?
        ORDER BY datetime(created_time) ASC
    '''
    async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
        async with conn.execute(query, (contact_id,)) as cursor:
            async for row in cursor:
                rows.append(row[0])
    return rows
