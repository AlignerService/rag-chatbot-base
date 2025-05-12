
from fastapi import FastAPI
from pydantic import BaseModel
import openai
import os
import json
import requests
import re
from typing import List
import threading

app = FastAPI()

CHUNKS_PATH = "./all_chunks.json"
MEMORY_PATH = "./customer_memory.json"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ZOHODESK_TOKEN = os.getenv("ZOHODESK_TOKEN")
ZOHODESK_API_URL = "https://desk.zoho.eu/api/v1"

MAX_THREAD_TOKENS = 1500
MAX_CHUNKS_TOKENS = 3000
MAX_TOTAL_TOKENS = 8000

memory_lock = threading.Lock()

def count_tokens(text: str) -> int:
    return len(text.split())

def truncate_tokens(text: str, max_tokens: int) -> str:
    words = text.split()
    return " ".join(words[:max_tokens])

def strip_html(text: str) -> str:
    return re.sub("<[^<]+?>", "", text.replace("\n", " ").replace("&nbsp;", " ")).strip()

def load_memory() -> dict:
    if not os.path.exists(MEMORY_PATH):
        return {}
    with open(MEMORY_PATH, "r") as f:
        return json.load(f)

def save_memory(data: dict):
    with memory_lock:
        with open(MEMORY_PATH, "w") as f:
            json.dump(data, f, indent=2)

with open(CHUNKS_PATH, "r") as f:
    CHUNKS = json.load(f)

class ChatRequest(BaseModel):
    ticketId: str = None
    question: str = None

def fetch_ticket_thread(ticket_id: str) -> (str, str):
    url = f"{ZOHODESK_API_URL}/tickets/{ticket_id}/threads"
    headers = {"Authorization": f"Zoho-oauthtoken {ZOHODESK_TOKEN}"}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return "", ""

    threads = response.json().get("data", [])
    contact_email = ""
    if threads and "contact" in threads[0]:
        contact_email = threads[0]["contact"].get("email", "")

    if not threads:
        return "", contact_email

    messages = []

    first = threads[0]
    messages.append(f"Initial message from customer: {strip_html(first.get('content', ''))}")

    for t in threads[-3:]:
        direction = t.get("direction")
        content = strip_html(t.get("content", ""))
        if direction == "in":
            messages.append(f"Customer follow-up: {content}")
        elif direction == "out":
            messages.append(f"Agent reply: {content}")

    joined = "\n".join(messages)
    return truncate_tokens(joined, MAX_THREAD_TOKENS), contact_email

def find_relevant_chunks(query: str, top_k: int = 10) -> List[str]:
    results = []
    for chunk in CHUNKS:
        score = query.lower().count(chunk["content"].lower()[:20])
        results.append((score, chunk["content"]))
    results.sort(reverse=True)
    selected = [text for _, text in results[:top_k]]

    collected = []
    tokens = 0
    for text in selected:
        t = count_tokens(text)
        if tokens + t > MAX_CHUNKS_TOKENS:
            break
        collected.append(text)
        tokens += t
    return collected

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    ticket_id = req.ticketId
    user_input = req.question or "Answer this based on the context."

    try:
        thread_context, customer_email = fetch_ticket_thread(ticket_id) if ticket_id else ("", "")
        top_chunks = find_relevant_chunks(thread_context or user_input)
        memory = load_memory()
        personal_context = memory.get(customer_email, "") if customer_email else ""

        # Sæt prompt sammen
        context_parts = [personal_context, thread_context, "\n---\n".join(top_chunks)]
        context = "\n\n".join([part for part in context_parts if part])
        total_tokens = count_tokens(context + user_input)
        if total_tokens > MAX_TOTAL_TOKENS:
            context = truncate_tokens(context, MAX_TOTAL_TOKENS - count_tokens(user_input))

        prompt = f"Based on the following context, give a helpful and accurate reply.\n\n{context}\n\nUser asked: {user_input}"

        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        chat_completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful dental assistant AI."},
                {"role": "user", "content": prompt},
            ]
        )
        reply = chat_completion.choices[0].message.content

        # Gem kort resumé af samtalen
        if customer_email:
            new_memory = memory.get(customer_email, "")
            summary = f"Ticket {ticket_id} handled with topic: {user_input}\nShort summary: {reply[:500]}"
            updated = f"{new_memory}\n\n{summary}".strip()
            memory[customer_email] = truncate_tokens(updated, 3000)
            save_memory(memory)

        return {"reply": reply}
    except Exception as e:
        return {"reply": "Beklager, der opstod en fejl i AI-assistenten. Prøv igen senere."}
