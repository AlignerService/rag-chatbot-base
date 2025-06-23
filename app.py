import os
import json
import sqlite3
import logging
import numpy as np
import asyncio
import tiktoken
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import openai
import faiss
import dropbox
import requests
import time
import threading

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Miljøvariabler
required_envs = [
    "OPENAI_API_KEY", "DROPBOX_CLIENT_ID", "DROPBOX_CLIENT_SECRET",
    "DROPBOX_REFRESH_TOKEN", "DROPBOX_DB_PATH"
]
missing_envs = [env for env in required_envs if not os.getenv(env)]
if missing_envs:
    raise RuntimeError(f"Missing required environment variables: {missing_envs}")

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DROPBOX_CLIENT_ID = os.getenv("DROPBOX_CLIENT_ID")
DROPBOX_CLIENT_SECRET = os.getenv("DROPBOX_CLIENT_SECRET")
DROPBOX_REFRESH_TOKEN = os.getenv("DROPBOX_REFRESH_TOKEN")
DROPBOX_DB_PATH = os.getenv("DROPBOX_DB_PATH")
LOCAL_DB_PATH = os.getenv("LOCAL_DB_PATH", "/tmp/rag.sqlite3")

EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4")
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0.2))

client = openai.OpenAI(api_key=OPENAI_API_KEY)

db_lock = asyncio.Lock()

# --- Thread-safe Dropbox Token Manager ---
class DropboxTokenManager:
    def __init__(self, client_id, client_secret, refresh_token):
        self.client_id = client_id
        self.client_secret = client_secret
        self.refresh_token = refresh_token
        self.access_token = None
        self.token_expires_at = 0
        self.lock = threading.Lock()

    def get_access_token(self):
        with self.lock:
            now = time.time()
            if self.access_token is None or now >= self.token_expires_at:
                self.refresh_access_token()
            return self.access_token

    def refresh_access_token(self):
        url = "https://api.dropbox.com/oauth2/token"
        data = {
            "grant_type": "refresh_token",
            "refresh_token": self.refresh_token,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        resp = requests.post(url, data=data)
        if resp.status_code == 200:
            token_data = resp.json()
            self.access_token = token_data["access_token"]
            expires_in = token_data.get("expires_in", 14400)
            self.token_expires_at = time.time() + expires_in - 60
            logging.info(f"Refreshed Dropbox access token, expires in {expires_in} sec")
        else:
            logging.error(f"Failed to refresh Dropbox token: {resp.text}")
            raise RuntimeError(f"Failed to refresh Dropbox token: {resp.text}")

token_manager = DropboxTokenManager(DROPBOX_CLIENT_ID, DROPBOX_CLIENT_SECRET, DROPBOX_REFRESH_TOKEN)

# Genbrug af Dropbox klient per kald
def get_dropbox_client():
    token = token_manager.get_access_token()
    return dropbox.Dropbox(token)

# --- Dropbox download/upload med genbrug og uden race via asyncio.Queue ---

upload_queue = asyncio.Queue()
upload_in_progress = False

async def upload_worker():
    global upload_in_progress
    while True:
        await upload_queue.get()
        try:
            client_dbx = get_dropbox_client()
            with open(LOCAL_DB_PATH, "rb") as f:
                client_dbx.files_upload(f.read(), DROPBOX_DB_PATH, mode=dropbox.files.WriteMode.overwrite)
            logging.info(f"Uploaded DB to Dropbox from {LOCAL_DB_PATH}")
        except Exception as e:
            logging.error(f"Error uploading DB to Dropbox: {e}")
        finally:
            upload_queue.task_done()
            upload_in_progress = False

def queue_db_upload():
    global upload_in_progress
    if not upload_in_progress:
        upload_in_progress = True
        asyncio.create_task(upload_queue.put(1))

def download_db_from_dropbox():
    try:
        client_dbx = get_dropbox_client()
        metadata, res = client_dbx.files_download(DROPBOX_DB_PATH)
        with open(LOCAL_DB_PATH, "wb") as f:
            f.write(res.content)
        logging.info(f"Downloaded DB from Dropbox to {LOCAL_DB_PATH}")
    except Exception as e:
        logging.error(f"Critical error downloading DB from Dropbox: {e}")
        raise RuntimeError("Critical error downloading DB from Dropbox")

# --- Init og start upload worker ---
download_db_from_dropbox()
asyncio.get_event_loop().create_task(upload_worker())

# --- Init DB som før ---
def init_db():
    try:
        with sqlite3.connect(LOCAL_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tickets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket_id TEXT,
                    question TEXT,
                    answer TEXT,
                    source TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
            logging.info("Database initialized successfully.")
    except Exception as e:
        logging.error(f"DB initialization error: {e}")
        raise

init_db()

# --- FAISS og metadata load med fejlstop ---
INDEX_FILE = os.getenv("FAISS_INDEX_FILE", "faiss.index")
METADATA_FILE = os.getenv("METADATA_FILE", "metadata.json")

try:
    index = faiss.read_index(INDEX_FILE)
    expected_dim = 1536
    if index.d != expected_dim:
        logging.error(f"FAISS index dimension mismatch: expected {expected_dim}, got {index.d}")
        raise RuntimeError("FAISS index dimension mismatch")
    logging.info(f"FAISS index loaded from {INDEX_FILE}")
except Exception as e:
    logging.error(f"Failed to load FAISS index: {e}")
    raise

try:
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    if len(metadata) != index.ntotal:
        logging.warning(f"Metadata length {len(metadata)} != FAISS index entries {index.ntotal}")
    logging.info(f"Metadata loaded from {METADATA_FILE}")
except Exception as e:
    logging.error(f"Failed to load metadata: {e}")
    raise

# --- Resten af koden: AnswerRequest, trim_context, get_top_chunks, generate_answer ---

tokenizer = tiktoken.get_encoding("cl100k_base")  # Instansér tokenizer ÉN gang

class AnswerRequest(BaseModel):
    ticketId: str = Field(..., max_length=100)
    question: str = Field(..., max_length=1000)

    @validator("question")
    def question_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("Question must not be empty")
        return v

def trim_context(context_chunks, max_tokens=6000):
    tokens_used = 0
    trimmed_chunks = []
    for chunk in context_chunks:
        if 'text' not in chunk:
            continue
        chunk_tokens = len(tokenizer.encode(chunk['text']))
        if tokens_used + chunk_tokens > max_tokens:
            break
        trimmed_chunks.append(chunk)
        tokens_used += chunk_tokens
    return trimmed_chunks

def get_top_chunks(question: str, top_k: int = 5):
    if index is None or not metadata:
        logging.warning("Index or metadata not loaded, cannot search.")
        return []

    try:
        response = asyncio.run(client.embeddings.acreate(input=[question], model=EMBEDDING_MODEL))
        if not response.data or len(response.data) == 0:
            logging.error("OpenAI embedding response empty")
            return []
        query_vector = response.data[0].embedding
    except Exception as e:
        logging.error(f"OpenAI embedding error: {e}")
        return []

    D, I = index.search(np.array([query_vector]).astype('float32'), top_k)
    result_chunks = []
    for idx in I[0]:
        if idx < len(metadata):
            entry = metadata[idx]
            if "text" in entry:
                result_chunks.append(entry)
    return result_chunks

async def generate_answer(question: str, context_chunks: list):
    if not context_chunks:
        return "Der findes ikke nok information til at besvare spørgsmålet. Skal jeg sende spørgsmålet videre til en klinisk ekspert?"

    trimmed_chunks = trim_context(context_chunks, max_tokens=6000)
    context = "\n\n---\n\n".join([chunk['text'] for chunk in trimmed_chunks])

    prompt = (
        "Du er Karin fra AlignerService, en erfaren klinisk rådgiver.\n"
        "Svar så informativt som muligt baseret på følgende kontekst.\n"
        "Hvis spørgsmålet kræver klinisk ekspertise ud over din kapacitet, så sig det tydeligt og tilbyd at involvere en klinisk rådgiver.\n\n"
        f"Kontekst:\n{context}\n\nSpørgsmål:\n{question}\n\nSvar:"
    )

    try:
        response = await client.chat.completions.acreate(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"OpenAI chat completion error: {e}")
        return f"Der opstod en fejl ved generering af svar: {e}"

def save_to_db_sync(ticket_id, question, answer_text):
    with db_lock:  # Trådsikkerhed på DB
        try:
            with sqlite3.connect(LOCAL_DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO tickets (ticket_id, question, answer, source)
                    VALUES (?, ?, ?, ?)
                ''', (ticket_id, question, answer_text, "RAG"))
                conn.commit()
            logging.info(f"Saved ticket {ticket_id} to database")
            # Queue upload, men vent ikke på at den er færdig
            asyncio.create_task(upload_queue.put(1))
        except Exception as e:
            logging.error(f"DB insert error: {e}")
            raise

upload_queue = asyncio.Queue()
upload_in_progress = False

async def upload_worker():
    global upload_in_progress
    while True:
        await upload_queue.get()
        try:
            client_dbx = get_dropbox_client()
            with open(LOCAL_DB_PATH, "rb") as f:
                client_dbx.files_upload(f.read(), DROPBOX_DB_PATH, mode=dropbox.files.WriteMode.overwrite)
            logging.info(f"Uploaded DB to Dropbox from {LOCAL_DB_PATH}")
        except Exception as e:
            logging.error(f"Error uploading DB to Dropbox: {e}")
        finally:
            upload_queue.task_done()
            upload_in_progress = False

def get_dropbox_client():
    token = token_manager.get_access_token()
    return dropbox.Dropbox(token)

@app.post("/api/answer")
async def answer(request: AnswerRequest):
    logging.info(f"Received question for ticketId {request.ticketId}")

    chunks = await asyncio.to_thread(get_top_chunks, request.question, top_k=5)
    answer_text = await generate_answer(request.question, chunks)

    try:
        await asyncio.to_thread(save_to_db_sync, request.ticketId, request.question, answer_text)
    except Exception as e:
        logging.error(f"DB insert error: {e}")
        raise HTTPException(status_code=500, detail="Database error")

    return {"answer": answer_text}

@app.on_event("startup")
async def startup_event():
    # Start upload worker på baggrund
    asyncio.create_task(upload_worker())

@app.get("/health")
async def health_check():
    try:
        with sqlite3.connect(LOCAL_DB_PATH) as conn:
            conn.execute("SELECT 1")
    except Exception as e:
        logging.error(f"Health check DB error: {e}")
        return {"status": "error", "detail": "DB access failed"}

    if not index:
        logging.error("Health check FAISS index not loaded")
        return {"status": "error", "detail": "FAISS index not loaded"}

    if not metadata:
        logging.error("Health check metadata not loaded")
        return {"status": "error", "detail": "Metadata not loaded"}

    return {"status": "ok"}
