import os
import json
import faiss
import numpy as np
import aiosqlite
import logging
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken

# --- Load environment and initialize OpenAI client ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Setup logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants ---
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
CHAT_MODEL     = os.getenv("OPENAI_CHAT_MODEL", "gpt-4")
TEMPERATURE    = float(os.getenv("OPENAI_TEMPERATURE", 0.2))
INDEX_FILE     = os.getenv("FAISS_INDEX_FILE", "faiss.index")
METADATA_FILE  = os.getenv("METADATA_FILE", "metadata.json")
LOCAL_DB_PATH  = os.getenv("LOCAL_DB_PATH", "/tmp/knowledge.sqlite")

# --- Tokenizer ---
tokenizer = tiktoken.get_encoding("cl100k_base")

def num_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

def count_and_log(name: str, text: str) -> int:
    toks = num_tokens(text)
    logger.info(f"[TokenUsage] {name}: {toks} tokens")
    return toks

async def get_customer_history(contact_id: str, limit: int = 5) -> list[str]:
    async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
        cursor = await conn.execute(
            "SELECT content FROM ticket_threads WHERE contact_id = ? ORDER BY created_time DESC LIMIT ?",
            (contact_id, limit)
        )
        rows = await cursor.fetchall()
    return [r[0] for r in reversed(rows)]

async def get_ticket_history(ticket_id: str, limit: int = 5) -> list[str]:
    async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
        cursor = await conn.execute(
            "SELECT content FROM ticket_threads WHERE ticket_id = ? ORDER BY created_time DESC LIMIT ?",
            (ticket_id, limit)
        )
        rows = await cursor.fetchall()
    return [r[0] for r in reversed(rows)]

async def search_and_answer(ticket_id: str, contact_id: str, question: str, top_k: int = 5) -> str:
    # 1) Load FAISS index and metadata
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # 2) Fetch histories
    ticket_hist = await get_ticket_history(ticket_id)
    ticket_text = "\n".join(ticket_hist)
    ticket_toks = count_and_log("TicketHistory", ticket_text)

    customer_hist = []
    if ticket_toks < 1000:
        customer_hist = await get_customer_history(contact_id)
    customer_text = "\n".join(customer_hist)
    customer_toks = count_and_log("CustomerHistory", customer_text)

    # 3) Embed question and search
    resp = client.embeddings.create(input=[question], model=EMBEDDING_MODEL)
    q_emb = np.array(resp.data[0].embedding, dtype=np.float32).reshape(1, -1)
    D, I = index.search(q_emb, top_k)
    rag_chunks = [metadata[i]['text'] for i in I[0] if i < len(metadata)]
    rag_text = "\n---\n".join(rag_chunks)
    rag_toks = count_and_log("RAGChunks", rag_text)

    # 4) Trim combined context
    MAX_CTX = 3000
    combined = ticket_hist + customer_hist + rag_chunks
    trimmed, used = [], 0
    for seg in combined:
        tok = num_tokens(seg)
        if used + tok > MAX_CTX:
            logger.info(f"[TokenUsage] Dropped segment of {tok} tokens due to budget overflow")
            break
        trimmed.append(seg)
        used += tok
    logger.info(f"[TokenUsage] CombinedContext: {used}/{MAX_CTX} tokens used")

    # 5) Build prompt
    prompt_lines = ["Du er tandlæge Helle Hatt fra AlignerService, en erfaren klinisk rådgiver."]
    if ticket_hist:
        prompt_lines.append("Tidligere samtaler (current ticket):")
        prompt_lines.extend(ticket_hist)
    if customer_hist:
        prompt_lines.append("Tidligere samtaler (andre tickets):")
        prompt_lines.extend(customer_hist)
    prompt_lines.append("Kontekst fra vores vidensbase:")
    prompt_lines.extend(rag_chunks)
    prompt_lines.append(f"Spørgsmål: {question}")
    prompt_lines.append("Svar:")
    prompt = "\n\n".join(prompt_lines)

    # 6) Generate answer
    chat = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role":"user","content":prompt}],
        temperature=TEMPERATURE,
        max_tokens=500
    )
    return chat.choices[0].message.content.strip()

# === CLI ===
if __name__ == "__main__":
    import asyncio
    tid = input("Ticket ID: ")
    cid = input("Contact ID: ")
    q   = input("Spørgsmål: ")
    ans = asyncio.run(search_and_answer(tid, cid, q))
    print("Svar:\n", ans)
