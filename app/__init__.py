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
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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

INDEX_FILE           = os.getenv("FAISS_INDEX_FILE", "faiss.index")
METADATA_FILE        = os.getenv("METADATA_FILE", "metadata.json")

# --- Globals for FAISS & metadata ---
index = None
metadata = None

# --- FastAPI app ---
app = FastAPI()

# --- CORS (så UI kan kalde fra hvor som helst) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# --- Database init & download with migration ---
async def download_db():
    try:
        dbx = await get_dropbox_client()
        md, res = await asyncio.to_thread(dbx.files_download, DROPBOX_DB_PATH)
        await asyncio.to_thread(lambda: open(LOCAL_DB_PATH, 'wb').write(res.content))
        logger.info("Downloaded DB from Dropbox")
    except Exception:
        logger.exception("Failed to download DB")

async def init_db():
    async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
        # 1) Opret tickets-tabellen
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

        # 2) Opret ticket_threads (uden de migrerede kolonner)
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS ticket_threads (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id    TEXT,
                contact_id   TEXT,
                sender       TEXT,
                content      TEXT
            );
        ''')

        # 3) Migration: tilføj contact_id & created_time hvis mangler
        cursor = await conn.execute("PRAGMA table_info(ticket_threads);")
        cols = [row[1] for row in await cursor.fetchall()]

        if 'contact_id' not in cols:
            await conn.execute("ALTER TABLE ticket_threads ADD COLUMN contact_id TEXT;")
            logger.info("Migrated ticket_threads: added contact_id column")

        if 'created_time' not in cols:
            await conn.execute("ALTER TABLE ticket_threads ADD COLUMN created_time TEXT;")
            logger.info("Migrated ticket_threads: added created_time column")

        await conn.commit()
        logger.info("DB initialized (with full migration)")

# --- Lazy Load FAISS & metadata ---
async def load_index_meta():
    global index, metadata
    try:
        index = await asyncio.to_thread(faiss.read_index, INDEX_FILE)
        metadata = await asyncio.to_thread(lambda: json.load(open(METADATA_FILE, 'r', encoding='utf-8')))
        logger.info("Loaded FAISS index and metadata")
    except Exception as e:
        logger.exception("Failed to load FAISS index/metadata")
        raise RuntimeError("Index load failed") from e

# --- Webhook routes (Zoho-token håndteres dér) ---
from app.webhook_integration import router as webhook_router
app.include_router(webhook_router)

# --- Request/Response Models ---
class AnswerRequest(BaseModel):
    ticketId:  str = Field(..., pattern=r'^[\w-]+$')
    contactId: str = Field(..., pattern=r'^[\w-]+$')
    question:  str

    @validator('question')
    def nonempty(cls, v):
        v = v.strip()
        if not v:
            raise ValueError('Question må ikke være tom')
        return html.escape(v)

class AnswerResponse(BaseModel):
    answer: str

from app.api_search_helpers import init_db_path, get_ticket_history, get_customer_history

# --- /api/answer endpoint ---
@app.post("/api/answer", response_model=AnswerResponse)
async def api_answer(req: AnswerRequest):
    global index, metadata
    if index is None or metadata is None:
        await load_index_meta()

    # 1) Ticket history
    ticket_hist = await get_ticket_history(req.ticketId)
    ticket_text = "\n".join(ticket_hist)
    count_and_log("TicketHistory", ticket_text)

    # 2) Customer history
    customer_hist = []
    if num_tokens(ticket_text) < 1000:
        customer_hist = await get_customer_history(req.contactId, exclude_ticket_id=req.ticketId)
    count_and_log("CustomerHistory", "\n".join(customer_hist))

    # 3) RAG-søgning
    emb = await async_client.embeddings.create(input=[req.question], model=EMBEDDING_MODEL)
    q_vec = np.array(emb.data[0].embedding, dtype=np.float32).reshape(1, -1)
    D, I = index.search(q_vec, 5)
    rag_chunks = [metadata[i]['text'] for i in I[0] if i < len(metadata)]
    count_and_log("RAGChunks", "\n---\n".join(rag_chunks))

    # 4) Trim context
    MAX_CTX = 3000
    used, ctx = 0, []
    for seg in ticket_hist + customer_hist + rag_chunks:
        tok = num_tokens(seg)
        if used + tok > MAX_CTX:
            break
        ctx.append(seg)
        used += tok

    # 5) Byg prompt
    parts = ["Du er tandlæge Helle Hatt fra AlignerService, en erfaren klinisk rådgiver."]
    if ticket_hist:
        parts += ["Tidligere samtaler (dette ticket):"] + ticket_hist
    if customer_hist:
        parts += ["Tidligere samtaler (andre tickets):"] + customer_hist
    parts += ["Faglig kontekst:"] + rag_chunks
    parts += [f"Spørgsmål: {req.question}", "Svar:"]
    prompt = "\n\n".join(parts)

    # 6) Chat completion med error‐handling
    try:
        chat = await async_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=500
        )
        answer = chat.choices[0].message.content.strip()
    except OpenAIError as e:
        logger.exception("OpenAI‐kald fejlede")
        raise HTTPException(status_code=502, detail="OpenAI‐tjeneste fejlede. Prøv igen senere.") from e

    # 7) Gem svar
    async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
        await conn.execute(
            "INSERT INTO tickets (ticket_id, contact_id, question, answer, source) VALUES (?, ?, ?, ?, 'RAG')",
            (req.ticketId, req.contactId, req.question, answer)
        )
        await conn.commit()
    await sync_mgr.queue()

    return {"answer": answer}

# --- Alias for UI på /answer ---
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
            """
            UPDATE tickets
            SET answer = ?, source = 'FINAL', created_at = CURRENT_TIMESTAMP
            WHERE ticket_id = ?
            """,
            (html.escape(log.finalAnswer), log.ticketId)
        )
        await conn.commit()
    await sync_mgr.queue()
    return {"status": "ok"}

# --- Healthcheck & Render uptime-check på HEAD / og GET / ---
@app.head("/", include_in_schema=False)
async def health_head():
    return JSONResponse(status_code=200, content=None)

@app.get("/", include_in_schema=False)
async def health_get():
    return {"status": "ok"}

# --- Startup & shutdown ---
@app.on_event("startup")
async def on_startup():
    # 1) download DB
    await download_db()
    # 2) init DB + migration
    await init_db()
    # 3) init api_search_helpers
    init_db_path(LOCAL_DB_PATH)
    # 4) (valgfrit) preload index/metadata for tidlig fejl
    try:
        await load_index_meta()
    except RuntimeError:
        pass
    logger.info("Startup complete")

@app.on_event("shutdown")
async def on_shutdown():
    await sync_mgr.queue()
    await asyncio.sleep(2)
