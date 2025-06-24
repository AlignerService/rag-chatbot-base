import html
import asyncio
import numpy as np
import tiktoken
import aiosqlite
from fastapi import FastAPI
from pydantic import BaseModel, Field, validator

# Core logic imported from app_core
from .app_core import (
    download_db,
    init_db,
    load_index_meta,
    sync_mgr,
    init_db_path,
    async_client,
    client,
    LOCAL_DB_PATH,
    INDEX_FILE,
    METADATA_FILE,
    EMBEDDING_MODEL,
    CHAT_MODEL,
    TEMPERATURE
)
from .webhook_integration import router as webhook_router
from .api_search_helpers import get_ticket_history, get_customer_history

# --- FastAPI App ---
app = FastAPI()
app.include_router(webhook_router)

# --- Tokenizer ---
tokenizer = tiktoken.get_encoding("cl100k_base")

def num_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

def count_and_log(name: str, text: str) -> None:
    toks = num_tokens(text)
    # Using asyncio-safe logging
    print(f"[TokenUsage] {name}: {toks} tokens")

# --- Models ---
class AnswerRequest(BaseModel):
    ticketId: str = Field(..., pattern=r'^[\w-]+$')
    contactId: str = Field(..., pattern=r'^[\w-]+$')
    question: str

    @validator("question")
    def nonempty(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Question must not be empty")
        return html.escape(v)

class AnswerResponse(BaseModel):
    answer: str

# --- Answer Endpoint ---
@app.post("/api/answer", response_model=AnswerResponse)
async def api_answer(req: AnswerRequest):
    # Ticket history
    ticket_hist = await get_ticket_history(req.ticketId)
    ticket_text = "\n".join(ticket_hist)
    count_and_log("TicketHistory", ticket_text)

    # Customer history if tokens allow
    customer_hist = []
    if num_tokens(ticket_text) < 1000:
        customer_hist = await get_customer_history(req.contactId, exclude_ticket_id=req.ticketId)
    cust_text = "\n".join(customer_hist)
    count_and_log("CustomerHistory", cust_text)

    # RAG search
    emb = await async_client.embeddings.create(input=[req.question], model=EMBEDDING_MODEL)
    q_emb = np.array(emb.data[0].embedding, dtype=np.float32).reshape(1, -1)
    idx = load_index_meta.__self__.index  # loaded index
    D, I = idx.search(q_emb, 5)
    # retrieve metadata
    meta = load_index_meta.__self__.metadata
    rag_chunks = [meta[i]['text'] for i in I[0] if i < len(meta)]
    rag_text = "\n---\n".join(rag_chunks)
    count_and_log("RAGChunks", rag_text)

    # Assemble context within limits
    MAX_CTX = 3000
    combined = ticket_hist + customer_hist + rag_chunks
    used = 0
    context = []
    for seg in combined:
        tok = num_tokens(seg)
        if used + tok > MAX_CTX:
            print(f"[TokenUsage] Dropped {tok} tokens segment due to overflow")
            break
        context.append(seg)
        used += tok
    print(f"[TokenUsage] TotalContext: {used}/{MAX_CTX} tokens")

    # Build prompt
    prompt = [
        "Du er tandlæge Helle Hatt fra AlignerService, en erfaren klinisk rådgiver."
    ]
    if ticket_hist:
        prompt.append("Tidligere samtaler (dette ticket):")
        prompt.extend(ticket_hist)
    if customer_hist:
        prompt.append("Tidligere samtaler (andre tickets):")
        prompt.extend(customer_hist)
    prompt.append("Faglig kontekst:")
    prompt.extend(rag_chunks)
    prompt.append(f"Spørgsmål: {req.question}")
    prompt.append("Svar:")
    prompt_str = "\n\n".join(prompt)

    # Generate answer
    chat = await async_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role":"user","content":prompt_str}],
        temperature=TEMPERATURE,
        max_tokens=500
    )
    answer = chat.choices[0].message.content.strip()

    # Save to DB
    async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
        await conn.execute(
            'INSERT INTO tickets (ticket_id, contact_id, question, answer, source) VALUES (?, ?, ?, ?, ?)',
            (req.ticketId, req.contactId, req.question, answer, 'RAG')
        )
        await conn.commit()
    await sync_mgr.queue()

    return {"answer": answer}

# --- Startup & Shutdown ---
@app.on_event("startup")
async def on_startup():
    init_db_path(LOCAL_DB_PATH)
    await download_db()
    await init_db()
    await load_index_meta()
    print("Startup complete")

@app.on_event("shutdown")
async def on_shutdown():
    await sync_mgr.queue()
    await asyncio.sleep(2)
