
from fastapi import FastAPI
from pydantic import BaseModel
import openai
import os
import json
import requests
import re
from typing import List

app = FastAPI()

# ---- RAG config ----
CHUNKS_PATH = "./all_chunks.json"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ZOHODESK_TOKEN = os.getenv("ZOHODESK_TOKEN")
ZOHODESK_API_URL = "https://desk.zoho.eu/api/v1"

# ---- Load chunks ----
with open(CHUNKS_PATH, "r") as f:
    CHUNKS = json.load(f)

# ---- Input model ----
class ChatRequest(BaseModel):
    ticketId: str = None
    question: str = None

# ---- Strip HTML tags ----
def strip_html(text: str) -> str:
    return re.sub("<[^<]+?>", "", text.replace("\n", " ").replace("&nbsp;", " ")).strip()

# ---- Get ZoHo Desk thread (prioritér første + sidste beskeder) ----
def fetch_ticket_thread(ticket_id: str) -> str:
    url = f"{ZOHODESK_API_URL}/tickets/{ticket_id}/threads"
    headers = {"Authorization": f"Zoho-oauthtoken {ZOHODESK_TOKEN}"}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return ""

    threads = response.json().get("data", [])
    if not threads:
        return ""

    messages = []

    # Første besked
    first = threads[0]
    messages.append(f"Initial message from customer: {strip_html(first.get('content', ''))}")

    # Seneste 2-3 beskeder
    for t in threads[-3:]:
        direction = t.get("direction")
        content = strip_html(t.get("content", ""))
        if direction == "in":
            messages.append(f"Customer follow-up: {content}")
        elif direction == "out":
            messages.append(f"Agent reply: {content}")

    return "\n".join(messages)

# ---- Find relevant chunks ----
def find_relevant_chunks(query: str, top_k: int = 3) -> List[str]:
    results = []
    for chunk in CHUNKS:
        score = query.lower().count(chunk["content"].lower()[:20])
        results.append((score, chunk["content"]))
    results.sort(reverse=True)
    return [text for _, text in results[:top_k]]

# ---- Chat endpoint ----
@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    ticket_id = req.ticketId
    user_input = req.question or "Answer this based on the context."

    thread_context = fetch_ticket_thread(ticket_id) if ticket_id else ""
    top_chunks = find_relevant_chunks(thread_context or user_input)

    context = thread_context + "\n\n" + "\n---\n".join(top_chunks)
    prompt = f"Based on the following information, respond helpfully and accurately.\n\n{context}\n\nUser asked: {user_input}"

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    chat_completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful dental assistant AI."},
            {"role": "user", "content": prompt},
        ]
    )

    return {"reply": chat_completion.choices[0].message.content}
