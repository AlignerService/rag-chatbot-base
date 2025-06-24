# app/app_core.py
import os
import json
import logging
import asyncio
import time
from datetime import datetime

import numpy as np
import faiss
import dropbox
import tiktoken
import aiohttp
import aiosqlite
from openai import OpenAI, AsyncOpenAI

# --- Load env (if you still need it here) ---
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# --- Settings & Constants ---
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL     = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
CHAT_MODEL          = os.getenv("OPENAI_CHAT_MODEL", "gpt-4")
TEMPERATURE         = float(os.getenv("OPENAI_TEMPERATURE", 0.2))

DROPBOX_CLIENT_ID     = os.getenv("DROPBOX_CLIENT_ID")
DROPBOX_CLIENT_SECRET = os.getenv("DROPBOX_CLIENT_SECRET")
DROPBOX_REFRESH_TOKEN = os.getenv("DROPBOX_REFRESH_TOKEN")
DROPBOX_DB_PATH       = os.getenv("DROPBOX_DB_PATH")
LOCAL_DB_PATH         = os.getenv("LOCAL_DB_PATH", "/tmp/knowledge.sqlite")

ZOHO_TOKEN_URL       = "https://accounts.zoho.eu/oauth/v2/token"
ZOHO_API_URL         = "https://desk.zoho.eu/api/v1"
TOKEN_CACHE_FILE     = os.getenv("ZOHO_TOKEN_CACHE", "token_cache.json")

INDEX_FILE           = os.getenv("FAISS_INDEX_FILE", "faiss.index")
METADATA_FILE        = os.getenv("METADATA_FILE", "metadata.json")

# --- OpenAI clients ---
client       = OpenAI(api_key=OPENAI_API_KEY)
async_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# --- Dropbox token manager & sync ---
class AsyncDropboxTokenManager:
    # ... copy your implementation from app.py ...
    pass

class DropboxSyncManager:
    # ... copy implementation ...
    pass

dropbox_token_mgr = AsyncDropboxTokenManager(DROPBOX_CLIENT_ID, DROPBOX_CLIENT_SECRET, DROPBOX_REFRESH_TOKEN)
sync_mgr = DropboxSyncManager()

async def download_db():
    # ... copy your download_db logic ...
    pass

async def init_db():
    # ... your aiosqlite init logic ...
    pass

async def load_index_meta():
    # should load into module‐level `index` and `metadata`
    global index, metadata
    # ... copy your faiss.read_index + json.load logic ...
    pass

def init_db_path(path: str):
    global LOCAL_DB_PATH
    LOCAL_DB_PATH = path
