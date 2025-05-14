
import sqlite3
import os

# Brug absolut sti til Dropbox-mappe
DB_PATH = "/Users/macpro/Dropbox/AlignerService/RAG:Database:aktiv/knowledge.sqlite"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_id TEXT UNIQUE,
            question TEXT,
            answer TEXT,
            source TEXT
        )
    ''')
    conn.commit()
    conn.close()

def insert_ticket(ticket_id, question, answer, source="ZoHo Ticket"):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO tickets (ticket_id, question, answer, source)
            VALUES (?, ?, ?, ?)
        ''', (ticket_id, question, answer, source))
        conn.commit()
    except sqlite3.IntegrityError:
        print(f"⚠️ Ticket med ID '{ticket_id}' findes allerede i databasen.")
    finally:
        conn.close()

if __name__ == "__main__":
    init_db()
    # Eksempel: indsæt en test-ticket
    insert_ticket("12345", "How do I perform IPR?", "Use Dentatus strips and avoid early stripping before stage 3.", "Test Input")
    print("✅ Ticket indsat.")
