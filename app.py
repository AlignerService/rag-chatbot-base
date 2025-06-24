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
TOKEN_CACHE_FILE     = os.getenv("ZOHO_TOKEN_CACHE", "token_cache.json")

INDEX_FILE           = os.getenv("FAISS_INDEX_FILE", "faiss.index")
METADATA_FILE        = os.getenv("METADATA_FILE", "metadata.json")

# --- FastAPI app ---
app = FastAPI()

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

# --- Include webhook routes ---
from app.webhook_integration import router as webhook_router
app.include_router(webhook_router)

# --- Search & Answer endpoint ---
class AnswerRequest(BaseModel):
    ticketId:  str = Field(..., pattern=r'^[\w-]+$')
    contactId: str = Field(..., pattern=r'^[\w-]+$')
    question:  str

    @validator("question")
    def nonempty(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Question must not be empty")
        return v

class AnswerResponse(BaseModel):
    answer: str

@app.post("/api/answer", response_model=AnswerResponse)
async def api_answer(req: AnswerRequest):
    # Load FAISS & metadata
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # Ticket history
    ticket_hist = await get_ticket_history(req.ticketId)
    ticket_text = "\n".join(ticket_hist)
    tk_toks = count_and_log("TicketHistory", ticket_text)

    # Customer history if space
    customer_hist = []
    if tk_toks < 1000:
        customer_hist = await get_customer_history(req.contactId)
    cust_text = "\n".join(customer_hist)
    ct_toks = count_and_log("CustomerHistory", cust_text)

    # RAG search
    emb_resp = client.embeddings.create(input=[req.question], model=EMBEDDING_MODEL)
    q_emb = np.array(emb_resp.data[0].embedding, dtype=np.float32).reshape(1, -1)
    D, I = index.search(q_emb, 5)
    rag_chunks = [metadata[i]['text'] for i in I[0] if i < len(metadata)]
    rag_text = "\n---\n".join(rag_chunks)
    rc_toks = count_and_log("RAGChunks", rag_text)

    # Trim
    MAX_CTX = 3000
    combined = ticket_hist + customer_hist + rag_chunks
    trimmed, used = [], 0
    for seg in combined:
        tok = num_tokens(seg)
        if used + tok > MAX_CTX:
            logger.info(f"[TokenUsage] Dropped segment of {tok} tokens due to overflow")
            break
        trimmed.append(seg)
        used += tok
    logger.info(f"[TokenUsage] TotalContext: {used}/{MAX_CTX} tokens")

    # Build prompt
    prompt_lines = ["Du er tandlæge Helle Hatt fra AlignerService, en erfaren klinisk rådgiver."]
    if ticket_hist:
        prompt_lines.append("Tidligere samtaler (dette ticket):")
        prompt_lines.extend(ticket_hist)
    if customer_hist:
        prompt_lines.append("Tidligere samtaler (andre tickets):")
        prompt_lines.extend(customer_hist)
    prompt_lines.append("Faglig kontekst:")
    prompt_lines.extend(rag_chunks)
    prompt_lines.append(f"Spørgsmål: {req.question}")
    prompt_lines.append("Svar:")
    prompt = "\n\n".join(prompt_lines)

    # Generate answer
    chat = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role":"user","content":prompt}],
        temperature=TEMPERATURE,
        max_tokens=500
    )

    # Save answer
    async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
        await conn.execute(
            'INSERT INTO tickets (ticket_id, contact_id, question, answer, source) VALUES (?, ?, ?, ?, ?)',
            (req.ticketId, req.contactId, req.question, chat.choices[0].message.content.strip(), 'RAG')
        )
        await conn.commit()
    await sync_mgr.queue()

    return {"answer": chat.choices[0].message.content.strip()}

# Helper imports
from app.api_search_helpers import get_ticket_history, get_customer_history

# --- Startup & Shutdown ---
@app.on_event("startup")
async def on_startup():
    await download_db()
    await init_db()
    await load_index_meta()
    logger.info("Startup complete")

@app.on_event("shutdown")
async def on_shutdown():
    await sync_mgr.queue()
    await asyncio.sleep(2)
