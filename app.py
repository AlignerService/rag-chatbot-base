import os
import json
import sqlite3
import logging
import numpy as np
import asyncio
import tiktoken
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, validator
import openai
import faiss
import dropbox
import requests
import time
import html
from datetime import datetime
import aiohttp

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Miljøvariabler ---
required_env_vars = [
    "OPENAI_API_KEY", "DROPBOX_CLIENT_ID", "DROPBOX_CLIENT_SECRET",
    "DROPBOX_REFRESH_TOKEN", "DROPBOX_DB_PATH",
    "ZOHO_CLIENT_ID", "ZOHO_CLIENT_SECRET", "ZOHO_REFRESH_TOKEN"
]
missing = [v for v in required_env_vars if not os.getenv(v)]
if missing:
    raise RuntimeError(f"Missing required environment variables: {missing}")

# OpenAI config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4")
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0.2))

# Dropbox config
DROPBOX_CLIENT_ID = os.getenv("DROPBOX_CLIENT_ID")
DROPBOX_CLIENT_SECRET = os.getenv("DROPBOX_CLIENT_SECRET")
DROPBOX_REFRESH_TOKEN = os.getenv("DROPBOX_REFRESH_TOKEN")
DROPBOX_DB_PATH = os.getenv("DROPBOX_DB_PATH")
LOCAL_DB_PATH = os.getenv("LOCAL_DB_PATH", "/tmp/knowledge.sqlite")

# Zoho config
ZOHO_CLIENT_ID = os.getenv("ZOHO_CLIENT_ID")
ZOHO_CLIENT_SECRET = os.getenv("ZOHO_CLIENT_SECRET")
ZOHO_REFRESH_TOKEN = os.getenv("ZOHO_REFRESH_TOKEN")
ZOHO_TOKEN_URL = "https://accounts.zoho.eu/oauth/v2/token"
ZOHO_API_URL = "https://desk.zoho.eu/api/v1"

# FAISS + Metadata config
INDEX_FILE = os.getenv("FAISS_INDEX_FILE", "faiss.index")
METADATA_FILE = os.getenv("METADATA_FILE", "metadata.json")

# --- FastAPI app ---
app = FastAPI()

# --- OpenAI async client ---
from openai import AsyncOpenAI
async_openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# --- Async Dropbox Token Manager ---
class AsyncDropboxTokenManager:
    def __init__(self, client_id, client_secret, refresh_token):
        self.client_id = client_id
        self.client_secret = client_secret
        self.refresh_token = refresh_token
        self.access_token = None
        self.token_expires_at = 0
        self._lock = asyncio.Lock()

    async def get_access_token(self):
        async with self._lock:
            now = time.time()
            if self.access_token is None or now >= self.token_expires_at:
                await self._refresh_access_token()
            return self.access_token

    async def _refresh_access_token(self):
        url = "https://api.dropbox.com/oauth2/token"
        data = {
            "grant_type": "refresh_token",
            "refresh_token": self.refresh_token,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data) as resp:
                if resp.status == 200:
                    token_data = await resp.json()
                    self.access_token = token_data["access_token"]
                    expires_in = token_data.get("expires_in", 14400)
                    self.token_expires_at = time.time() + expires_in - 60
                    logger.info(f"Refreshed Dropbox access token")
                else:
                    error_text = await resp.text()
                    logger.error(f"Failed to refresh Dropbox token: {error_text}")
                    raise RuntimeError(f"Failed to refresh Dropbox token: {error_text}")

dropbox_token_manager = AsyncDropboxTokenManager(DROPBOX_CLIENT_ID, DROPBOX_CLIENT_SECRET, DROPBOX_REFRESH_TOKEN)

async def get_dropbox_client_async():
    token = await dropbox_token_manager.get_access_token()
    return dropbox.Dropbox(token)

# --- DropboxSyncManager with async queue and lock ---
class DropboxSyncManager:
    def __init__(self):
        self._upload_queue = asyncio.Queue()
        self._upload_in_progress = False
        self._lock = asyncio.Lock()

    async def queue_upload(self):
        async with self._lock:
            if not self._upload_in_progress:
                self._upload_in_progress = True
                await self._upload_queue.put(time.time())

    async def upload_worker(self):
        while True:
            await self._upload_queue.get()
            try:
                await self._upload_to_dropbox()
            except Exception as e:
                logger.error(f"Dropbox upload failed: {e}")
            finally:
                async with self._lock:
                    self._upload_in_progress = False
                self._upload_queue.task_done()

    async def _upload_to_dropbox(self):
        await asyncio.to_thread(self._sync_upload)

    def _sync_upload(self):
        # Synchronous upload - run in thread
        dbx = dropbox.Dropbox(dropbox_token_manager.access_token)
        with open(LOCAL_DB_PATH, "rb") as f:
            dbx.files_upload(f.read(), DROPBOX_DB_PATH, mode=dropbox.files.WriteMode.overwrite)
        logger.info("Uploaded SQLite DB to Dropbox")

sync_manager = DropboxSyncManager()

# --- Download DB from Dropbox ---
async def download_db_from_dropbox_async():
    try:
        dbx = await get_dropbox_client_async()
        metadata, res = dbx.files_download(DROPBOX_DB_PATH)
        with open(LOCAL_DB_PATH, "wb") as f:
            f.write(res.content)
        logger.info(f"Downloaded DB from Dropbox to {LOCAL_DB_PATH}")
    except Exception as e:
        logger.error(f"Failed to download DB from Dropbox: {e}")
        raise RuntimeError("Failed to download DB from Dropbox")

# --- Init SQLite DB schema ---
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
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ticket_threads (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket_id TEXT,
                    sender TEXT,
                    content TEXT,
                    time TEXT,
                    UNIQUE(ticket_id, time)
                )
            ''')
            conn.commit()
            logger.info("SQLite DB initialized")
    except Exception as e:
        logger.error(f"Failed to initialize DB: {e}")
        raise

init_db()

# --- Load FAISS and metadata ---
try:
    index = faiss.read_index(INDEX_FILE)
    expected_dim = 1536
    if index.d != expected_dim:
        raise RuntimeError(f"FAISS index dimension mismatch: expected {expected_dim}, got {index.d}")
    logger.info(f"FAISS index loaded from {INDEX_FILE}")
except Exception as e:
    logger.error(f"Failed to load FAISS index: {e}")
    raise

try:
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    if len(metadata) != index.ntotal:
        logger.warning(f"Metadata length {len(metadata)} != FAISS index entries {index.ntotal}")
    logger.info(f"Metadata loaded from {METADATA_FILE}")
except Exception as e:
    logger.error(f"Failed to load metadata: {e}")
    raise

tokenizer = tiktoken.get_encoding("cl100k_base")

# --- Request models ---
class AnswerRequest(BaseModel):
    ticketId: str = Field(..., max_length=100, pattern=r'^[a-zA-Z0-9_-]+$')  # <-- rettet her
    question: str = Field(..., max_length=2000)

    @validator("question")
    def question_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("Question must not be empty")
        v = html.escape(v.strip())
        return v

class UpdateRequest(BaseModel):
    ticketId: str

# --- Helper functions for RAG ---
def trim_context(chunks, max_tokens=6000):
    tokens_used = 0
    trimmed = []
    for chunk in chunks:
        if 'text' not in chunk:
            continue
        chunk_tokens = len(tokenizer.encode(chunk['text']))
        if tokens_used + chunk_tokens > max_tokens:
            break
        trimmed.append(chunk)
        tokens_used += chunk_tokens
    return trimmed

def get_top_chunks(question: str, top_k: int = 5):
    if index is None or not metadata:
        logger.warning("Index or metadata not loaded")
        return []

    try:
        response = openai.ChatCompletion.create(
            model=EMBEDDING_MODEL,
            messages=[{"role": "user", "content": question}]
        )
        if not response.data or len(response.data) == 0:
            logger.error("Empty OpenAI embedding response")
            return []
        query_vector = response.data[0].embedding
    except Exception as e:
        logger.error(f"OpenAI embedding error: {e}")
        return []

    D, I = index.search(np.array([query_vector]).astype('float32'), top_k)
    results = []
    for idx in I[0]:
        if idx < len(metadata):
            entry = metadata[idx]
            if "text" in entry:
                results.append(entry)
    return results

async def generate_answer(question: str, context_chunks: list):
    if not context_chunks:
        return "Der findes ikke nok information til at besvare spørgsmålet. Skal jeg sende spørgsmålet videre til en klinisk ekspert?"

    trimmed = trim_context(context_chunks)
    context_text = "\n\n---\n\n".join([chunk['text'] for chunk in trimmed])

    prompt = (
        "Du er Karin fra AlignerService, en erfaren klinisk rådgiver.\n"
        "Svar så informativt som muligt baseret på følgende kontekst.\n"
        "Hvis spørgsmålet kræver klinisk ekspertise ud over din kapacitet, så sig det tydeligt og tilbyd at involvere en klinisk rådgiver.\n\n"
        f"Kontekst:\n{context_text}\n\nSpørgsmål:\n{question}\n\nSvar:"
    )

    try:
        response = await async_openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI chat completion error: {e}")
        return f"Der opstod en fejl ved generering af svar: {e}"

# --- Zoho token refresh ---
def get_zoho_access_token():
    payload = {
        "refresh_token": ZOHO_REFRESH_TOKEN,
        "client_id": ZOHO_CLIENT_ID,
        "client_secret": ZOHO_CLIENT_SECRET,
        "grant_type": "refresh_token"
    }
    try:
        resp = requests.post(ZOHO_TOKEN_URL, data=payload, timeout=10)
        resp.raise_for_status()
        token_data = resp.json()
        access_token = token_data.get("access_token")
        if not access_token:
            raise Exception("No access token in Zoho response")
        return access_token
    except Exception as e:
        logger.error(f"Zoho token refresh error: {e}")
        raise HTTPException(status_code=500, detail="Failed to refresh Zoho token")

# --- Zoho API calls and DB sync ---
async def get_ticket_thread_async(ticket_id):
    token = get_zoho_access_token()
    headers = {"Authorization": f"Zoho-oauthtoken {token}"}
    url = f"{ZOHO_API_URL}/tickets/{ticket_id}/conversations"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=10) as resp:
                resp.raise_for_status()
                return await resp.json()
    except Exception as e:
        logger.error(f"Zoho API error getting ticket thread: {e}")
        return {}

async def store_ticket_thread_async(ticket_id, thread_data):
    import aiosqlite
    async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS ticket_threads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id TEXT,
                sender TEXT,
                content TEXT,
                time TEXT,
                UNIQUE(ticket_id, time)
            )
        ''')
        values = []
        for item in thread_data.get("data", []):
            sender = (item.get("fromEmail") or item.get("sender") or "unknown")[:255]
            content = html.unescape(item.get("content") or "").strip()[:10000]
            timestamp = item.get("createdTime") or datetime.utcnow().isoformat()
            values.append((ticket_id, sender, content, timestamp))
        await conn.executemany(
            "INSERT OR IGNORE INTO ticket_threads (ticket_id, sender, content, time) VALUES (?, ?, ?, ?)",
            values
        )
        await conn.commit()
    await sync_manager.queue_upload()

# --- REST endpoints ---
@app.get("/")
def read_root():
    return {"message": "✅ FastAPI is working."}

@app.get("/health")
async def health_check():
    import aiosqlite
    try:
        async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
            await conn.execute("SELECT 1")
    except Exception as e:
        logger.error(f"Health check DB error: {e}")
        return {"status": "error", "detail": "DB access failed"}

    if not index:
        logger.error("Health check FAISS index not loaded")
        return {"status": "error", "detail": "FAISS index not loaded"}

    if not metadata:
        logger.error("Health check metadata not loaded")
        return {"status": "error", "detail": "Metadata not loaded"}

    return {"status": "ok"}

@app.get("/tickets")
async def get_ticket(ticket_id: str):
    import aiosqlite
    if not ticket_id or not ticket_id.strip().isalnum():
        raise HTTPException(status_code=400, detail="Invalid ticketId format")
    try:
        async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
            async with conn.execute("SELECT sender, content, time FROM ticket_threads WHERE ticket_id = ? ORDER BY time ASC", (ticket_id,)) as cursor:
                rows = await cursor.fetchall()
        if not rows:
            raise HTTPException(status_code=404, detail="No data found for this ticket")
        return [{"sender": r[0], "content": r[1], "time": r[2]} for r in rows]
    except Exception as e:
        logger.error(f"Error reading ticket: {e}")
        raise HTTPException(status_code=500, detail="Failed to read ticket data")

@app.post("/update_ticket")
async def update_ticket(req: Request):
    try:
        body = await req.json()
        ticket_id = body.get("ticketId")
        if not ticket_id or not ticket_id.strip().isalnum():
            raise HTTPException(status_code=400, detail="Invalid ticketId format")
        thread = await get_ticket_thread_async(ticket_id)
        if not thread.get("data"):
            raise HTTPException(status_code=404, detail="No conversations found for ticket")
        await store_ticket_thread_async(ticket_id, thread)
        return {"status": "Ticket thread saved", "ticketId": ticket_id}
    except Exception as e:
        logger.error(f"Exception in /update_ticket: {e}")
        raise HTTPException(status_code=500, detail="Failed in update_ticket")

@app.post("/api/answer")
async def api_answer(request: AnswerRequest):
    logger.info(f"Received AI question for ticketId {request.ticketId}")
    chunks = await asyncio.to_thread(get_top_chunks, request.question, top_k=5)
    answer = await generate_answer(request.question, chunks)
    import aiosqlite
    try:
        async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
            await conn.execute('''
                INSERT INTO tickets (ticket_id, question, answer, source)
                VALUES (?, ?, ?, ?)
            ''', (request.ticketId, request.question, answer, "RAG"))
            await conn.commit()
        await sync_manager.queue_upload()
        logger.info(f"Saved AI answer for ticket {request.ticketId}")
    except Exception as e:
        logger.error(f"Failed saving AI answer: {e}")
        raise HTTPException(status_code=500, detail="Database error")
    return {"answer": answer}

# --- Middleware: Timeout ---
@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    try:
        return await asyncio.wait_for(call_next(request), timeout=30.0)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Request timeout")

# --- Startup & Shutdown ---
@app.on_event("startup")
async def startup():
    try:
        await download_db_from_dropbox_async()
        logger.info("Downloaded DB at startup successfully")
    except Exception as e:
        logger.error(f"Failed to download DB at startup: {e}")
        # optionally raise here to stop the app if critical
        # raise

    asyncio.create_task(sync_manager.upload_worker())

    try:
        await async_openai_client.embeddings.create(input=["test"], model=EMBEDDING_MODEL)
        logger.info("OpenAI connection validated")
    except Exception as e:
        logger.error(f"OpenAI connection failed: {e}")
        raise

@app.on_event("shutdown")
async def shutdown():
    await sync_manager.queue_upload()
    await asyncio.sleep(2)
