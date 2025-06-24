import os
import json
import aiosqlite
import html
import asyncio
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI

# --- Load environment and initialize OpenAI clients ---
load_dotenv()
sync_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Paths & settings ---
LOCAL_DB_PATH = os.getenv("LOCAL_DB_PATH", "/tmp/knowledge.sqlite")
TOP_K = int(os.getenv("RAG_TOP_K", 5))
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4")
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0.2))

# --- Utility for escaping and truncating ---
def sanitize(text: str, max_len: int = 10000) -> str:
    return html.escape(text)[:max_len]

# --- Embedding and FAISS search (sync) ---
def get_embedding(text: str):
    resp = sync_client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
    return resp.data[0].embedding

async def get_top_chunks(question: str, index, metadata) -> list:
    emb = get_embedding(question)
    D, I = index.search([emb], TOP_K)
    return [metadata[i] for i in I[0] if i < len(metadata)]

# --- Async DB update logic ---
async def update_knowledge_db(ticket_id: str, question: str, answer: str, source: str = "RAG"):
    async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
        # Ensure tables exist
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id TEXT,
                question TEXT,
                answer TEXT,
                source TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        # Insert record
        await conn.execute(
            "INSERT INTO tickets (ticket_id, question, answer, source) VALUES (?, ?, ?, ?)",
            (ticket_id, sanitize(question), sanitize(answer), source)
        )
        await conn.commit()

# --- CLI example usage ---
if __name__ == "__main__":
    import faiss
    # Load index & metadata
    index = faiss.read_index(os.getenv("FAISS_INDEX_FILE", "faiss.index"))
    with open(os.getenv("METADATA_FILE", "metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)

    tid = input("Ticket ID: ")
    q   = input("Question: ")

    # Generate answer via RAG (simplified)
    from rag_answer import get_rag_answer
    ans = get_rag_answer(q)

    # Update DB
    asyncio.run(update_knowledge_db(tid, q, ans))
    print("Saved to DB:", tid)
