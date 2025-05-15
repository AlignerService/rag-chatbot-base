from fastapi import FastAPI, Request
from pydantic import BaseModel
import sqlite3
import dropbox
import os

app = FastAPI()

# Dropbox upload opsætning
DROPBOX_TOKEN = "indsæt_din_token_her"
dbx = dropbox.Dropbox(DROPBOX_TOKEN)
DROPBOX_DB_PATH = "/knowledge.sqlite"

# Midlertidig filsti lokalt (Render skriver her, inden den uploader til Dropbox)
TEMP_LOCAL_DB_PATH = "/tmp/knowledge.sqlite"

class UpdateRequest(BaseModel):
    ticket_id: str
    question: str
    answer: str

@app.get("/")
def read_root():
    return {"message": "RAG API is live."}

@app.post("/update")
def update_knowledge_db(req: UpdateRequest):
    try:
        # Opret eller åbn lokal SQLite-database i tmp
        conn = sqlite3.connect(TEMP_LOCAL_DB_PATH)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id TEXT,
                question TEXT,
                answer TEXT
            )
        ''')

        cursor.execute('''
            INSERT INTO knowledge (ticket_id, question, answer)
            VALUES (?, ?, ?)
        ''', (req.ticket_id, req.question, req.answer))

        conn.commit()
        conn.close()

        # Upload databasen til Dropbox
        with open(TEMP_LOCAL_DB_PATH, "rb") as f:
            dbx.files_upload(f.read(), DROPBOX_DB_PATH, mode=dropbox.files.WriteMode.overwrite)

        return {"message": "Ticket saved and uploaded to Dropbox."}

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
