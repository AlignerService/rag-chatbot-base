from fastapi import FastAPI, Request
from pydantic import BaseModel
import sqlite3
import dropbox
import os

app = FastAPI()

# Dropbox upload opsætning
DROPBOX_TOKEN = "sl.u.AFujA7mJB1XBWnlTIwgPpQrvvJsXNMk6gTdvAolcXBnhzWD3-McJlWMZD0TffJ9ukRExAsKdtKNu39heFhiNMh4jjxTpUwiwi3ZtJLFDUa_qtQs6IX9nZj9OiHJj2WsliOSdpVjTsNE7KEfaCxqeGHCDJ6pkiYD3U9Bhz9MPNUTvNptngET47WXVfqwPGXNIMs7tGJbeJeFFs8jIP3Vm2pVex0COvdMsZmygb95KMinMWVBxgTNeAt4vX0UGaUk1Cw0gkMEN7U777uHgUIFdJdP4Izs543-bo9BtSnqu9dKy9XMKVAsW_Mby-029cg7ct68zd3_65FgFkIdcadKXpbZvAYynWoupIkl_nE3iSUVaUKYDrbhqFW9LZAtBuDJ40IV5wmXVVUhNVpEdDFfHyUfyaNBF495x6zE7kwCmmuZHgTyNDwJjcRmH5vBqA1XEFDzCKCMM2pA5oNy4PQd-cDb7DB0PNuxs5yMW2Ocexp6qkwzIKfG5qmZ950iImw-YKS2cpF0ZyYxIufDAnxfRCtSFq6io2s0PEK46cIkg6DJ4MLfQHNvh-GgD9lmnwE3z-yKtGHMVky-GzoaFnoORtTFD1xvwMg9o_AfhbVWrDDYC2ncuXBeLyfaOWLNff47J_E9nkybjlA9ezgm-JLiO3ACezvFe0JwWM66PbJnMVQAraY39xTnrYktqC178h0kNwYluWqQdnhnmTlIxS0oEa5aQB_rUTIyKXi5fpqJ409LIu43cBKkoFoxuxqmxZew2Bzpw-0UGxvsti894Qk1JuiN704aWs4XxxHYS_MqLtseMINoybB68gscXjWePu7Q3mcpGoikNYmTcXaUZOnA51yvWp0uWHnmNZ2qno-hBNIaEppzuOiAb_lTUrX5LzsCri_2sWZaZXDIjphvcbVdGuDz8cWop3LgfR9A2kHVpAu2Mrzjm723VFtBzEpZxuX6Ae9Oh2rv6tqne8i23n4qitOxUWvG92SGHUPfYIucBsmOo4rgX-WPDGltdguzXRisttLo4vAqVP0BYVnL9qROjqP91IZBm_qq9g-ga5tfR-tXql7VYABBLbpeuDM6AVmQI3N63Nzc64QcocNCwq-NVUOANv0FHF2KmOa7JnuqSFccswH7vzu_hFW9IvOJ4zhv3CuHreSp1kCaZLgJbyqUkwf_OFCNJWyo1KACecCphrzP7DTrgpZKigGEbJ9g1SWFHxmEq9iblIyzIf10C1jEo_o9GQrjsviGw5NoPw0PUn1SA-XXoJgik_JKf0o1XzFo6O1jY_gZmcCpD67mbw5h9Zb12iACWeSCaENM3bgMbfdbLih8LmV-1-2Py-fiAFBXYM3g"
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
