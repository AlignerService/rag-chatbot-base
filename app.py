
import os
import sqlite3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import requests

app = FastAPI()

# Define request format
class RequestData(BaseModel):
    ticketId: str
    question: str

# Get environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ZOHO_CLIENT_ID = os.getenv("ZOHO_CLIENT_ID")
ZOHO_CLIENT_SECRET = os.getenv("ZOHO_CLIENT_SECRET")
ZOHO_REFRESH_TOKEN = os.getenv("ZOHO_REFRESH_TOKEN")
ZOHO_ORGID = os.getenv("ZOHO_ORGID")
DB_PATH = "knowledge.sqlite"

# Set OpenAI key
openai.api_key = OPENAI_API_KEY

def refresh_zoho_token():
    url = "https://accounts.zoho.eu/oauth/v2/token"
    params = {
        "refresh_token": ZOHO_REFRESH_TOKEN,
        "client_id": ZOHO_CLIENT_ID,
        "client_secret": ZOHO_CLIENT_SECRET,
        "grant_type": "refresh_token"
    }
    response = requests.post(url, params=params)
    if response.status_code == 200 and "access_token" in response.json():
        return response.json()["access_token"]
    return None

def fetch_ticket_description(ticket_id: str, access_token: str):
    url = f"https://desk.zoho.eu/api/v1/tickets/{ticket_id}"
    headers = {
        "Authorization": f"Zoho-oauthtoken {access_token}",
        "orgId": ZOHO_ORGID
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json().get("description", "")
    return ""

def search_context(question: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT content FROM chunks WHERE content LIKE ?", (f"%{question}%",))
    rows = cursor.fetchall()
    conn.close()
    return "
---
".join([row[0] for row in rows])

@app.post("/answer")
async def generate_answer(payload: RequestData):
    try:
        access_token = refresh_zoho_token()
        if not access_token:
            return {"reply": "Kunne ikke opdatere ZoHo access token."}

        ticket_text = fetch_ticket_description(payload.ticketId, access_token)
        if not ticket_text:
            return {"reply": "Kunne ikke hente ticket-data fra ZoHo."}

        context = search_context(payload.question)

        messages = [
            {"role": "system", "content": "Du er en hjælpsom AI-assistent for AlignerService."},
            {"role": "user", "content": f"Spørgsmål: {payload.question}

Relevant kontekst:
{context}

Tidligere besked fra kunden:
{ticket_text}"}
        ]

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=messages
        )

        return {"reply": response.choices[0].message.content}

    except Exception as e:
        return {"reply": f"Fejl ved generering af svar: {str(e)}"}
