import os
import sqlite3
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DB_PATH = "knowledge.sqlite"

class Payload(BaseModel):
    question: str
    ticketId: str

def get_context_from_db(question: str) -> str:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT content FROM chunks")
    chunks = [row[0] for row in cursor.fetchall()]
    conn.close()

    # Simple keyword matching for demo purposes
    matched_chunks = [chunk for chunk in chunks if question.lower() in chunk.lower()]
    return "
---
".join(matched_chunks[:5])

def refresh_zoho_token():
    refresh_token = os.getenv("ZOHO_REFRESH_TOKEN")
    client_id = os.getenv("ZOHO_CLIENT_ID")
    client_secret = os.getenv("ZOHO_CLIENT_SECRET")
    redirect_uri = os.getenv("ZOHO_REDIRECT_URI", "https://www.google.com")

    response = requests.post(
        "https://accounts.zoho.eu/oauth/v2/token",
        params={
            "refresh_token": refresh_token,
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "refresh_token",
            "redirect_uri": redirect_uri
        }
    )
    response.raise_for_status()
    return response.json().get("access_token")

def get_ticket_data(ticket_id: str) -> str:
    org_id = os.getenv("ZOHO_ORGID")
    access_token = os.getenv("ZOHO_ACCESS_TOKEN")

    headers = {
        "Authorization": f"Zoho-oauthtoken {access_token}",
        "orgId": org_id
    }

    url = f"https://desk.zoho.eu/api/v1/tickets/{ticket_id}"
    response = requests.get(url, headers=headers)
    if response.status_code == 401:
        access_token = refresh_zoho_token()
        headers["Authorization"] = f"Zoho-oauthtoken {access_token}"
        os.environ["ZOHO_ACCESS_TOKEN"] = access_token  # Optional: Update in env

        response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json().get("subject", "") + "
" + response.json().get("description", "")

@app.post("/answer")
def generate_answer(payload: Payload):
    try:
        ticket_data = get_ticket_data(payload.ticketId)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fejl ved hentning af ZoHo ticket: {e}")

    context = get_context_from_db(payload.question)
    try:
        messages = [
            {"role": "system", "content": "Du er en hjælpsom AI assistent for en virksomhed, der yder support til tandlæger om clear aligner behandlinger."},
            {"role": "user", "content": f"Spørgsmål: {payload.question}

Ticket data:
{ticket_data}

Relevant kontekst:
{context}"}
        ]
        completion = openai.chat.completions.create(model="gpt-4", messages=messages)
        return {"reply": completion.choices[0].message.content.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fejl ved OpenAI-kald: {e}")