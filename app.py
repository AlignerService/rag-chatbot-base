import os
import requests
from fastapi import FastAPI, Request
from pydantic import BaseModel
from openai import OpenAI
from typing import List

# FastAPI app
app = FastAPI()

# Zoho API credentials from environment
ZOHO_ACCESS_TOKEN = os.getenv("ZOHO_ACCESS_TOKEN")
ZOHO_API_BASE = "https://www.zohoapis.eu"

# OpenAI setup
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

class ChatRequest(BaseModel):
    ticketId: str
    question: str

def get_ticket_thread(ticket_id: str) -> str:
    url = f"{ZOHO_API_BASE}/desk/v1/tickets/{ticket_id}/threads"
    headers = {"Authorization": f"Zoho-oauthtoken {ZOHO_ACCESS_TOKEN}"}
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        return "Fejl i hentning af ticket-tråd."

    data = response.json()
    if "data" not in data:
        return "Ingen tråde fundet."

    messages = []
    for thread in data["data"]:
        content = thread.get("content", "")
        direction = thread.get("direction", "")
        if direction == "in":
            messages.append(f"Customer: {content}")
        elif direction == "out":
            messages.append(f"Agent: {content}")
        else:
            messages.append(content)

    return "\n\n".join(messages)

@app.post("/chat")
def chat_endpoint(request_data: ChatRequest):
    thread_text = get_ticket_thread(request_data.ticketId)

    try:
        chat_completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful support assistant for a company called AlignerService. You summarize and answer based on customer support threads."},
                {"role": "user", "content": f"Thread:
{thread_text}

Question: {request_data.question}"}
            ]
        )
        return {"reply": chat_completion.choices[0].message.content}
    except Exception as e:
        return {"error": str(e)}
