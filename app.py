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
aiohttp
import aiosqlite
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, validator
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Required env vars ---
required_env_vars = [
    "OPENAI_API_KEY",
    "DROPBOX_CLIENT_ID",
    "DROPBOX_CLIENT_SECRET",
    "DROPBOX_REFRESH_TOKEN",
    "DROPBOX_DB_PATH",
    "ZOHO_CLIENT_ID",
    "ZOHO_CLIENT_SECRET",
    "ZOHO_REFRESH_TOKEN",
]
missing = [v for v in required_env_vars if not os.getenv(v)]
if missing:
    raise RuntimeError(f"Missing required environment variables: {missing}")

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

INDEX_FILE           = os.getenv("FAISS_INDEX_FILE", "faiss.index")
METADATA_FILE        = os.getenv("METADATA_FILE", "metadata.json")

# --- FastAPI app ---
app = FastAPI()

# --- OpenAI clients ---
client       = OpenAI(api_key=OPENAI_API_KEY)
async_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# --- Tokenizer ---
tokenizer = tiktoken.get_encoding("cl100k_base")

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
            now = time.time()
            if not self.access_token or now >= self.expires_at:
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
                self.expires_at   = time.time() + tk.get("expires_in",14400) - 60
                logger.info("Refreshed Dropbox token")

dropbox_token_mgr = AsyncDropboxTokenManager(
    DROPBOX_CLIENT_ID, DROPBOX_CLIENT_SECRET, DROPBOX_REFRESH_TOKEN
)

async def get_dropbox_client():
    token = await dropbox_token_mgr.get_access_token()
    return dropbox.Dropbox(token)

# --- DB sync manager ---
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
            with open(LOCAL_DB_PATH,'rb') as f:
                dbx.files_upload(f.read(), DROPBOX_DB_PATH,
                                 mode=dropbox.files.WriteMode.overwrite)
            logger.info("Uploaded DB to Dropbox")
        except Exception:
            logger.exception("Dropbox upload failed")

sync_mgr = DropboxSyncManager()

# --- DB init & Dropbox download ---
async def download_db():
    try:
        dbx = await get_dropbox_client()
        md, res = await asyncio.to_thread(dbx.files_download, DROPBOX_DB_PATH)
        with open(LOCAL_DB_PATH,'wb') as f:
            f.write(res.content)
        logger.info("Downloaded DB from Dropbox")
    except Exception:
        logger.exception("Failed to download DB")

async def init_db():
    async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
        await conn.execute('''
        CREATE TABLE IF NOT EXISTS tickets (...);
        ''')
        await conn.execute('''
        CREATE TABLE IF NOT EXISTS ticket_threads (...);
        ''')
        await conn.commit()
        logger.info("DB initialized locally")

# --- Load FAISS & metadata ---
async def load_index_meta():
    global index, metadata
    idx = await asyncio.to_thread(faiss.read_index, INDEX_FILE)
    if idx.d != 1536:
        raise RuntimeError("Dimension mismatch")
    index = idx
    with open(METADATA_FILE,'r',encoding='utf-8') as f:
        metadata = json.load(f)
    logger.info("Loaded FAISS and metadata")

# --- RAG helpers ---
def trim_context(chunks, max_tokens=6000):
    tokens_used = 0; trimmed=[]
    for c in chunks:
        ctok = len(tokenizer.encode(c.get('text','')))
        if tokens_used+ctok>max_tokens: break
        trimmed.append(c); tokens_used+=ctok
    return trimmed

async def get_embedding(text):
    resp = await async_client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
    return resp.data[0].embedding

async def top_chunks(q, k=5):
    vec = await get_embedding(q)
    D,I = await asyncio.to_thread(index.search, np.array([vec],dtype='float32'), k)
    return [metadata[i] for i in I[0] if i<len(metadata)]

async def generate_answer(q,chunks):
    if not chunks:
        return "Ingen info. Skal vi involvere en klinisk ekspert?"
    ctx = "\n\n---\n\n".join([c['text'] for c in trim_context(chunks)])
    prompt = f"Du er Karin...\nKontekst:\n{ctx}\nSpørgsmål:\n{q}\nSvar:"
    res = await async_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role":"user","content":prompt}],
        temperature=TEMPERATURE, max_tokens=300
    )
    return res.choices[0].message.content.strip()

# --- Models ---
class AnswerRequest(BaseModel):
    ticketId: str = Field(..., max_length=100, pattern=r'^[\w-]+$')
    question: str = Field(..., max_length=2000)
    @validator('question')
    def nonempty(cls,v): return html.escape(v.strip()) if v.strip() else (_ for _ in ()).throw(ValueError('Question must not be empty'))

class UpdateRequest(BaseModel):
    ticketId: str = Field(..., max_length=100, pattern=r'^[\w-]+$')

# --- Endpoints ---
@app.get("/")
def root(): return {"message":"FastAPI is up"}

@app.get("/health")
async def health():
    try:
        async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
            await conn.execute("SELECT 1")
    except: return {"status":"error","detail":"DB fail"}
    if not index or not metadata:
        return {"status":"error","detail":"Index/meta missing"}
    return {"status":"ok"}

@app.post("/update_ticket")
async def update_ticket(req: UpdateRequest):
    # Zoho fetch & store
    
@app.post("/api/answer")
async def api_answer(req: AnswerRequest):
    chunks = await top_chunks(req.question)
    ans    = await generate_answer(req.question,chunks)
    # Save to DB & Dropbox
    return {"answer":ans}

# --- Startup/Shutdown ---
@app.on_event("startup")
async def on_start():
    await download_db(); await init_db(); await load_index_meta();
    await async_client.embeddings.create(input=["test"], model=EMBEDDING_MODEL)
    logger.info("Startup complete")

@app.on_event("shutdown")
async def on_shutdown():
    await sync_mgr.queue()
    await asyncio.sleep(2)
