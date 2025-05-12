
from fastapi import FastAPI, Request
from pydantic import BaseModel
import os
import requests
import json
from typing import List
from openai import OpenAI
from jiter import embed_texts, load_chunks, find_relevant_chunks

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ChatRequest(BaseModel):
    ticketId: str
    question: str = "Kan du give mig et overblik over denne sag?"

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # Trin 1: Hent tråd fra ZoHo
    zoho_token = os.getenv("ZOHO_TOKEN")
    zoho_orgid = os.getenv("ZOHO_ORGID")

    ticket_url = f"https://desk.zoho.eu/api/v1/tickets/{request.ticketId}/threads"
    headers = {
        "Authorization": f"Zoho-oauthtoken {zoho_token}",
        "orgId": zoho_orgid
    }

    zoho_response = requests.get(ticket_url, headers=headers)
    if zoho_response.status_code != 200:
        return {"error": "ZoHo API-fejl", "details": zoho_response.text}

    threads = zoho_response.json().get("data", [])

    # Trin 2: Kombinér indholdet fra tråden
    full_thread_text = ""
    for t in threads:
        if t.get("direction") == "in":
            full_thread_text += f"Kunde: {t.get('content', '')}\n"
        elif t.get("direction") == "out":
            full_thread_text += f"Support: {t.get('content', '')}\n"

    # Trin 3: Find relevante chunks
    all_chunks = load_chunks("all_chunks.json")
    relevant_chunks = find_relevant_chunks(request.question + "\n" + full_thread_text, all_chunks, top_k=6)

    # Trin 4: Send prompt til OpenAI
    context = "\n---\n".join([chunk['content'] for chunk in relevant_chunks])
    prompt = f"{context}\n\nSAG:\n{full_thread_text}\n\nSPØRGSMÅL:\n{request.question}"

    chat_completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Du er en venlig og erfaren supportmedarbejder hos AlignerService. Du hjælper tandlæger og klinikpersonale med spørgsmål vedr. behandling med clear aligners."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )

    answer = chat_completion.choices[0].message.content.strip()
    return {"reply": answer}
