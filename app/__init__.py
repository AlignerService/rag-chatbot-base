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
from fastapi import FastAPI, HTTPException, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from openai import OpenAI, AsyncOpenAI, OpenAIError
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
    "ZOHO_CLIENT_ID", "ZOHO_CLIENT_SECRET", "ZOHO_REFRESH_TOKEN",
    "RAG_BEARER_TOKEN",
]
missing = [v for v in required if not os.getenv(v)]
if missing:
    raise RuntimeError(f"Missing required env vars: {missing}")

# --- Settings ---
OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL      = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
CHAT_MODEL           = os.getenv("OPENAI_CHAT_MODEL", "gpt-4")
TEMPERATURE          = float(os.getenv("OPENAI_TEMPERATURE", 0.2))

DROPBOX_CLIENT_ID     = os.getenv("DROPBOX_CLIENT_ID")
DROPBOX_CLIENT_SECRET = os.getenv("DROPBOX_CLIENT_SECRET")
DROPBOX_REFRESH_TOKEN = os.getenv("DROPBOX_REFRESH_TOKEN")
DROPBOX_DB_PATH       = os.getenv("DROPBOX_DB_PATH")
LOCAL_DB_PATH         = os.getenv("LOCAL_DB_PATH", "/tmp/knowledge.sqlite")

INDEX_FILE           = os.getenv("FAISS_INDEX_FILE", "faiss.index")
METADATA_FILE        = os.getenv("METADATA_FILE", "metadata.json")

RAG_BEARER_TOKEN     = os.getenv("RAG_BEARER_TOKEN")

# --- FastAPI app ---
app = FastAPI()

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Security (Bearer token) ---
bearer_scheme = HTTPBearer()

def require_rag_token(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
):
    if credentials.scheme.lower() != "bearer" or credentials.credentials != RAG_BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing RAG token")
    return True

# --- OpenAI clients ---
client       = OpenAI(api_key=OPENAI_API_KEY)
async_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# --- Tokenizer ---
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

# --- Dropbox Sync Manager ---
class DropboxSyncManager:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._task = None

    async def queue(self):
        async with self._lock:
            if not self._task or self._task.done():
                self._task = asyncio.create_task(self._upload())

    async def _upload(self):
        try:
            dbx = await get_dropbox_client()
            data = await asyncio.to_thread(lambda: open(LOCAL_DB_PATH, 'rb').read())
            await asyncio.to_thread(
                dbx.files_upload,
                data,
                DROPBOX_DB_PATH,
                mode=dropbox.files.WriteMode.overwrite
            )
            logger.info("Uploaded DB to Dropbox")
        except Exception:
            logger.exception("Dropbox upload failed")

sync_mgr = DropboxSyncManager()

# --- Database init, schema + migrations ---
async def init_db():
    async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
        # Create tickets table
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
        # Create ticket_threads table
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS ticket_threads (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id     TEXT,
                contact_id    TEXT,
                sender        TEXT,
                content       TEXT,
                created_time  TEXT,
                UNIQUE(ticket_id, created_time)
            );
        ''')
        # Migrations (add missing columns if any)
        cursor = await conn.execute("PRAGMA table_info(ticket_threads);")
        cols = {row[1] for row in await cursor.fetchall()}
        if 'contact_id' not in cols:
            await conn.execute("ALTER TABLE ticket_threads ADD COLUMN contact_id TEXT;")
            logger.info("Migrated ticket_threads: added contact_id")
        if 'created_time' not in cols:
            await conn.execute("ALTER TABLE ticket_threads ADD COLUMN created_time TEXT;")
            logger.info("Migrated ticket_threads: added created_time")
        cursor = await conn.execute("PRAGMA table_info(tickets);")
        cols = {row[1] for row in await cursor.fetchall()}
        if 'contact_id' not in cols:
            await conn.execute("ALTER TABLE tickets ADD COLUMN contact_id TEXT;")
            logger.info("Migrated tickets: added contact_id")
        if 'source' not in cols:
            await conn.execute("ALTER TABLE tickets ADD COLUMN source TEXT;")
            logger.info("Migrated tickets: added source")
        if 'created_at' not in cols:
            await conn.execute("ALTER TABLE tickets ADD COLUMN created_at TIMESTAMP;")
            logger.info("Migrated tickets: added created_at")
        await conn.commit()
        logger.info("DB initialized (with migrations)")

# --- Download DB from Dropbox ---
async def download_db():
    try:
        dbx = await get_dropbox_client()
        md, res = await asyncio.to_thread(dbx.files_download, DROPBOX_DB_PATH)
        await asyncio.to_thread(lambda: open(LOCAL_DB_PATH, 'wb').write(res.content))
        logger.info("Downloaded DB from Dropbox")
    except Exception:
        logger.exception("Failed to download DB")

# --- Lazy load FAISS & metadata ---
index = None
metadata = None
async def load_index_meta():
    global index, metadata
    try:
        index = await asyncio.to_thread(faiss.read_index, INDEX_FILE)
        metadata = await asyncio.to_thread(lambda: json.load(open(METADATA_FILE, 'r', encoding='utf-8')))
        logger.info("Loaded FAISS index and metadata")
    except Exception as e:
        logger.exception("Failed to load FAISS index/metadata")
        raise RuntimeError("Index load failed") from e

# --- Include other routers ---
from app.webhook_integration import router as webhook_router
app.include_router(webhook_router)

# **Her er den RETTE import** af dine search helpers:
from .api_search_helpers import init_db_path, get_ticket_history, get_customer_history

# --- Request/Response Models ---
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

class LogRequest(BaseModel):
    ticketId:    str
    finalAnswer: str

# --- API Endpoints ---
@app.post(
    "/api/answer",
    response_model=AnswerResponse,
    dependencies=[Depends(require_rag_token)]
)
async def api_answer(req: AnswerRequest):
    # ... resten af din api_answer uændret ...

@app.post(
    "/answer",
    response_model=AnswerResponse,
    dependencies=[Depends(require_rag_token)]
)
async def alias_answer(req: AnswerRequest = Body(...)):
    return await api_answer(req)

@app.post(
    "/update_ticket",
    dependencies=[Depends(require_rag_token)]
)
async def update_ticket(log: LogRequest):
    # ... resten af update_ticket uændret ...

# --- Healthcheck ---
@app.head("/", include_in_schema=False)
async def health_head():
    return JSONResponse(status_code=200, content=None)

@app.get("/", include_in_schema=False)
async def health_get():
    return {"status": "ok"}

# --- Startup & Shutdown ---
@app.on_event("startup")
async def on_startup():
    await download_db()
    await init_db()
    init_db_path(LOCAL_DB_PATH)   # <<<<<< Her kalder vi nu den relative import
    try:
        await load_index_meta()
    except RuntimeError:
        pass
    logger.info("Startup complete")

@app.on_event("shutdown")
async def on_shutdown():
    await sync_mgr.queue()
    await asyncio.sleep(2)
