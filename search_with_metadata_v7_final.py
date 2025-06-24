import os
import json
import faiss
import numpy as np
import aiosqlite
from dotenv import load_dotenv
from openai import OpenAI

def load_env():
    load_dotenv()
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === Setup OpenAI v1 client ===
from openai import OpenAI
import dotenv
dotenv.load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === Constants ===
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
INDEX_FILE      = os.getenv("FAISS_INDEX_FILE", "faiss.index")
METADATA_FILE   = os.getenv("METADATA_FILE", "metadata.json")
LOCAL_DB_PATH   = os.getenv("LOCAL_DB_PATH", "/tmp/knowledge.sqlite")

async def get_customer_history(contact_id: str, max_msgs: int = 10) -> list[str]:
    async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
        cursor = await conn.execute(
            "SELECT content FROM ticket_threads WHERE contact_id = ? ORDER BY created_time DESC LIMIT ?",
            (contact_id, max_msgs)
        )
        rows = await cursor.fetchall()
    # reverse for chronological order
    return [r[0] for r in reversed(rows)]

async def get_ticket_history(ticket_id: str, max_msgs: int = 10) -> list[str]:
    async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
        cursor = await conn.execute(
            "SELECT content FROM ticket_threads WHERE ticket_id = ? ORDER BY created_time DESC LIMIT ?",
            (ticket_id, max_msgs)
        )
        rows = await cursor.fetchall()
    return [r[0] for r in reversed(rows)]

from tiktoken import get_encoding

tokenizer = get_encoding("cl100k_base")

def num_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

async def search_and_answer(ticket_id: str, contact_id: str, question: str, top_k: int = 5) -> str:
    # 1) Load FAISS and metadata
    index = faiss.read_index(INDEX_FILE)
    metadata = json.load(open(METADATA_FILE, "r", encoding="utf-8"))

    # 2) Fetch histories
    ticket_hist = await get_ticket_history(ticket_id)
    customer_hist = []
    # combine only if tokens allow
    hist_tokens = sum(num_tokens(m) for m in ticket_hist)
    if hist_tokens < 1000:
        # add customer-wide history
        customer_hist = await get_customer_history(contact_id)
    # build context
    context = ticket_hist + customer_hist

    # 3) Get embedding for question
    resp = client.embeddings.create(input=[question], model=EMBEDDING_MODEL)
    q_emb = np.array(resp.data[0].embedding, dtype=np.float32).reshape(1, -1)

    # 4) FAISS search
    D, I = index.search(q_emb, top_k)
    rag_chunks = [metadata[idx]["text"] for idx in I[0] if idx < len(metadata)]

    # 5) Trim combined context + rag to max tokens
    combined = context + rag_chunks
    trimmed, used = [], 0
    for seg in combined:
        tok = num_tokens(seg)
        if used + tok <= 3000:
            trimmed.append(seg)
            used += tok
        else:
            break

    # 6) Build prompt
    prompt = ["Du er tandlæge Helle Hatt fra AlignerService, en erfaren klinisk rådgiver."]
    if context:
        prompt.append("Tidligere samtaler:")
        prompt.extend([f"- {c}" for c in context])
    prompt.append("Kontekst fra vores videnkilder:")
    prompt.extend([f"- {c}" for c in rag_chunks])
    prompt.append(f"Spørgsmål: {question}")
    prompt.append("Svar:")
    prompt_text = "\n".join(prompt)

    # 7) Generate answer
    chat = client.chat.completions.create(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4"),
        messages=[{"role":"user","content":prompt_text}],
        temperature=float(os.getenv("OPENAI_TEMPERATURE", 0.2)),
        max_tokens=500
    )
    return chat.choices[0].message.content.strip()

if __name__ == "__main__":
    import asyncio
    tid = input("Ticket ID: ")
    cid = input("Contact ID: ")
    q   = input("Spørgsmål: ")
    ans = asyncio.run(search_and_answer(tid, cid, q))
    print("Svar:\n", ans)
