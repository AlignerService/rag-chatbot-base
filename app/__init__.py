import os
import json
import logging
import asyncio
import time
import html
from datetime import datetime

import numpy as np
import faiss
import dropbox
import tiktoken
import aiohttp
import aiosqlite
from fastapi import FastAPI, Body
from pydantic import BaseModel, Field, validator
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Required env vars ---
required = [
    "OPENAI_API_KEY",
    "DROPBOX_CLIENT_ID", "DROPBOX_CLIENT_SECRET", "DROPBOX_REFRESH_TOKEN", "DROPBOX_DB_PATH",
    "ZOHO_CLIENT_ID", "ZOHO_CLIENT_SECRET", "ZOHO_REFRESH_TOKEN"
]
missing = [v for v in required if not os.getenv(v)]
if missing:
    raise RuntimeError(f"Missing required env vars: {missing}")

# --- Settings ---
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL     = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
CHAT_MODEL          = os.getenv("OPENAI_CHAT_MODEL", "gpt-4")
TEMPERATURE         = float(os.getenv("OPENAI_TEMPERATURE", 0.2))

DROPBOX_CLIENT_ID     = os.getenv("DROPBOX_CLIENT_ID")
DROPBOX_CLIENT_SECRET = os.getenv("DROPBOX_CLIENT_SECRET")
DROPBOX_REFRESH_TOKEN = os.getenv("DROPBOX_REFRESH_TOKEN")
DROPBOX_DB_PATH       = os.getenv("DROPBOX_DB_PATH")
LOCAL_DB_PATH         = os.getenv("LOCAL_DB_PATH", "/tmp/knowledge.sqlite")

ZOHO_CLIENT_ID       = os.getenv("ZOHO_CLIENT_ID")
ZOHO_CLIENT_SECRET   = os.getenv("ZOHO_CLIENT_SECRET")
ZOHO_REFRESH_TOKEN   = os.getenv("ZOHO_REFRESH_TOKEN")
ZOHO_TOKEN_URL       = "https://accounts.zoho.eu/oauth/v2/token"
ZOHO_API_URL         = "https://desk.zoho.eu/api/v1"
TOKEN_CACHE_FILE     = os.getenv("ZOHO_TOKEN_CACHE", "token_cache.json")

INDEX_FILE           = os.getenv("FAISS_INDEX_FILE", "faiss.index")
METADATA_FILE        = os.getenv("METADATA_FILE", "metadata.json")

# --- Globals for FAISS & metadata ---
index = None
metadata = None

# --- FastAPI app ---
app = FastAPI()

# --- OpenAI clients ---
client       = OpenAI(api_key=OPENAI_API_KEY)
async_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# --- Tokenizer helpers ---
tokenizer = tiktoken.get_encoding("cl100k_base")
def num_tokens(text: str) -> int:
    return len(tokenizer.encode(text))
def count_and_log(name: str, text: str) -> int:
    toks = num_tokens(text)
    logger.info(f"[TokenUsage] {name}: {toks} tokens")
    return toks

# --- Dropbox Token Manager ---
class AsyncDropboxTokenManager:
    def __init__(self, client_id, client_secret, refresh_token):
        self.client_id     = client_id
        self.client_secret = client_secret
        self.refresh_token = refresh_token
        self.access_token  = None
        self.expires_at    = 0
        self._lock         = asyncio.Lock()

    async def get_access_token(self):
        async with self._lock:
            if not self.access_token or time.time() >= self.expires_at:
                await self._refresh()
            return self.access_token

    async def _refresh(self):
        url = "https://api.dropbox.com/oauth2/token"
        data = {
            "grant_type":    "refresh_token",
            "refresh_token": self.refresh_token,
            "client_id":     self.client_id,
            "client_secret": self.client_secret,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data) as resp:
                resp.raise_for_status()
                tk = await resp.json()
                self.access_token = tk["access_token"]
                self.expires_at   = time.time() + tk.get("expires_in", 14400) - 60
                logger.info("Refreshed Dropbox token")

dropbox_token_mgr = AsyncDropboxTokenManager(
    DROPBOX_CLIENT_ID, DROPBOX_CLIENT_SECRET, DROPBOX_REFRESH_TOKEN
)

async def get_dropbox_client():
    token = await dropbox_token_mgr.get_access_token()
    return dropbox.Dropbox(token)

# --- Dropbox Sync Manager with retry/back-off ---
class DropboxSyncManager:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._task = None

    async def queue(self):
        async with self._lock:
            if not self._task or self._task.done():
                self._task = asyncio.create_task(self._upload())

    async def _upload(self):
        backoff = 1
        for attempt in range(3):
            try:
                dbx = await get_dropbox_client()
                with open(LOCAL_DB_PATH, 'rb') as f:
                    dbx.files_upload(f.read(), DROPBOX_DB_PATH,
                                     mode=dropbox.files.WriteMode.overwrite)
                logger.info("Uploaded DB to Dropbox")
                return
            except Exception:
                logger.exception(f"Dropbox upload failed, retry in {backoff}s")
                await asyncio.sleep(backoff)
                backoff *= 2

sync_mgr = DropboxSyncManager()

# --- Database init & download with migration & back-filling ---
async def download_db():
    backoff = 1
    for attempt in range(3):
        try:
            dbx = await get_dropbox_client()
            md, res = await asyncio.to_thread(dbx.files_download, DROPBOX_DB_PATH)
            with open(LOCAL_DB_PATH, 'wb') as f:
                f.write(res.content)
            logger.info("Downloaded DB from Dropbox")
            return
        except Exception:
            logger.exception(f"Failed to download DB, retry in {backoff}s")
            await asyncio.sleep(backoff)
            backoff *= 2

async def init_db():
    async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
        # 1) Create tickets
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS tickets (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id    TEXT,
                contact_id   TEXT,
                question     TEXT,
                answer       TEXT,
                source       TEXT,
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        ''')
        # 2) Create ticket_threads without new columns
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS ticket_threads (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id    TEXT,
                contact_id   TEXT,
                sender       TEXT,
                content      TEXT
            );
        ''')
        # 3) Migrate: add created_time with default
        cursor = await conn.execute("PRAGMA table_info(ticket_threads);")
        cols = [row[1] for row in await cursor.fetchall()]
        if 'created_time' not in cols:
            await conn.execute(
                "ALTER TABLE ticket_threads ADD COLUMN created_time TEXT DEFAULT CURRENT_TIMESTAMP;"
            )
            logger.info("Migrated ticket_threads: added created_time column")
        # 4) Backfill any NULLs
        await conn.execute(
            "UPDATE ticket_threads SET created_time = CURRENT_TIMESTAMP WHERE created_time IS NULL;"
        )
        # 5) Unique constraint can’t be ALTERed—log a reminder
        logger.info("Ensure UNIQUE(ticket_id,created_time) on ticket_threads manually if needed")
        await conn.commit()
        logger.info("DB initialized (with migration)")

# --- Lazy–load FAISS index & metadata with error-handling ---
async def load_index_meta():
    global index, metadata
    try:
        index = await asyncio.to_thread(faiss.read_index, INDEX_FILE)
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        logger.info("Loaded FAISS index and metadata")
    except Exception as e:
        logger.error("Failed to load FAISS index or metadata: %s", e)
        raise

# --- Health-check root endpoint ---
@app.get("/")
async def root():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}

# --- Models & RAG-endpoint ---
class AnswerRequest(BaseModel):
    ticketId:  str = Field(..., pattern=r'^[\w-]+$')
    contactId: str = Field(..., pattern=r'^[\w-]+$')
    question:  str

    @validator('question')
    def nonempty(cls, v):
        v = v.strip()
        if not v:
            raise ValueError('Question must not be empty')
        return html.escape(v)

class AnswerResponse(BaseModel):
    answer: str

from app.api_search_helpers import get_ticket_history, get_customer_history

@app.post("/api/answer", response_model=AnswerResponse)
async def api_answer(req: AnswerRequest):
    # Lazy load FAISS
    global index, metadata
    if index is None or metadata is None:
        await load_index_meta()

    # 1) Ticket history
    ticket_hist = await get_ticket_history(req.ticketId)
    count_and_log("TicketHistory", "\n".join(ticket_hist))

    # 2) Customer history
    customer_hist = []
    if num_tokens("\n".join(ticket_hist)) < 1000:
        customer_hist = await get_customer_history(
            req.contactId, exclude_ticket_id=req.ticketId
        )
    count_and_log("CustomerHistory", "\n".join(customer_hist))

    # 3) RAG search
    emb = await async_client.embeddings.create(
        input=[req.question], model=EMBEDDING_MODEL
    )
    q_vec = np.array(emb.data[0].embedding, dtype=np.float32).reshape(1, -1)
    D, I = index.search(q_vec, 5)
    rag_chunks = [metadata[i]['text'] for i in I[0] if i < len(metadata)]
    count_and_log("RAGChunks", "\n---\n".join(rag_chunks))

    # 4) Assemble context
    MAX_CTX = 3000
    used, ctx = 0, []
    for seg in ticket_hist + customer_hist + rag_chunks:
        tok = num_tokens(seg)
        if used + tok > MAX_CTX:
            break
        ctx.append(seg)
        used += tok

    # 5) Build prompt
    parts = ["Du er tandlæge Helle Hatt fra AlignerService, en erfaren klinisk rådgiver."]
    if ticket_hist:
        parts += ["Tidligere samtaler (dette ticket):"] + ticket_hist
    if customer_hist:
        parts += ["Tidligere samtaler (andre tickets):"] + customer_hist
    parts += ["Faglig kontekst:"] + rag_chunks
    parts += [f"Spørgsmål: {req.question}", "Svar:"]
    prompt = "\n\n".join(parts)

    # 6) Chat completion
    chat = await async_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        max_tokens=500
    )
    answer = chat.choices[0].message.content.strip()

    # 7) Save to DB
    async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
        await conn.execute(
            "INSERT INTO tickets (ticket_id, contact_id, question, answer, source) "
            "VALUES (?, ?, ?, ?, 'RAG')",
            (req.ticketId, req.contactId, req.question, answer)
        )
        await conn.commit()
    await sync_mgr.queue()

    return {"answer": answer}

# --- Alias for UI (so fetch("/answer") stadig virker) ---
@app.post("/answer", response_model=AnswerResponse)
async def alias_answer(req: AnswerRequest = Body(...)):
    return await api_answer(req)

# --- /update_ticket endpoint ---
class LogRequest(BaseModel):
    ticketId:    str
    finalAnswer: str

@app.post("/update_ticket")
async def update_ticket(log: LogRequest):
    async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
        await conn.execute(
            "UPDATE tickets "
            "SET answer = ?, source = 'FINAL', created_at = CURRENT_TIMESTAMP "
            "WHERE ticket_id = ?",
            (html.escape(log.finalAnswer), log.ticketId)
        )
        await conn.commit()
    await sync_mgr.queue()
    return {"status": "ok"}

# --- Startup & shutdown ---
@app.on_event("startup")
async def on_startup():
    # 1) Hent DB fra Dropbox (med retry)
    await download_db()
    # 2) Init og migrér DB-skema
    await init_db()
    # 3) Nu kan api_search_helpers bruge den nye DB-path
    from app.api_search_helpers import init_db_path
    init_db_path(LOCAL_DB_PATH)
    logger.info("Startup complete")

@app.on_event("shutdown")
async def on_shutdown():
    await sync_mgr.queue()
    await asyncio.sleep(2)
