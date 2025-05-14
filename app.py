import os
import sqlite3
import openai
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# OpenAI config
openai.api_key = os.getenv("OPENAI_API_KEY")

# Zoho OAuth config
ZOHO_CLIENT_ID = os.getenv("ZOHO_CLIENT_ID")
ZOHO_CLIENT_SECRET = os.getenv("ZOHO_CLIENT_SECRET")
ZOHO_REFRESH_TOKEN = os.getenv("ZOHO_REFRESH_TOKEN")
ZOHO_ORGID = os.getenv("ZOHO_ORGID")
ZOHO_ACCESS_TOKEN = os.getenv("ZOHO_ACCESS_TOKEN")

DB_PATH = "knowledge.sqlite"


class Payload(BaseModel):
    ticketId: str
    question: str


def refresh_zoho_access_token():
    url = "https://accounts.zoho.eu/oauth/v2/token"
    params = {
        "refresh_token": ZOHO_REFRESH_TOKEN,
        "client_id": ZOHO_CLIENT_ID,
        "client_secret": ZOHO_CLIENT_SECRET,
        "grant_type": "refresh_token",
    }
    response = requests.post(url, params=params)
    if response.status_code == 200 and "access_token" in response.json():
        new_token = response.json()["access_token"]
        os.environ["ZOHO_ACCESS_TOKEN"] = new_token
        return new_token
    else:
        raise HTTPException(status_code=500, detail="Failed to refresh ZoHo access token.")


def get_ticket_thread(ticket_id: str):
    token = os.getenv("ZOHO_ACCESS_TOKEN")
    if not token:
        token = refresh_zoho_access_token()
    url = f"https://desk.zoho.eu/api/v1/tickets/{ticket_id}/threads"
    headers = {
        "Authorization": f"Zoho-oauthtoken {token}",
        "orgId": ZOHO_ORGID,
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 401:
        token = refresh_zoho_access_token()
        headers["Authorization"] = f"Zoho-oauthtoken {token}"
        response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    raise HTTPException(status_code=response.status_code, detail="Failed to fetch ZoHo ticket thread.")


def fetch_relevant_context(question: str, db_path: str, top_k: int = 5) -> str:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT text FROM chunks")
    chunks = [row[0] for row in cursor.fetchall()]
    conn.close()
    return "\n\n".join(chunks[:top_k])


@app.post("/answer")
def answer(payload: Payload):
    try:
        ticket_data = get_ticket_thread(payload.ticketId)
        context = fetch_relevant_context(payload.question, DB_PATH)
        prompt = [
            {"role": "system", "content": "Du er en hjælpsom AI-assistent for AlignerService."},
            {
                "role": "user",
                "content": f"Spørgsmål: {payload.question}\n\nRelevant kontekst: {context}\n\nSvar venligst på spørgsmålet så præcist og hjælpsomt som muligt."
            }
        ]
        completion = openai.chat.completions.create(
            model="gpt-4",
            messages=prompt
        )
        return {"reply": completion.choices[0].message.content.strip()}
    except Exception as e:
        return {"reply": f"Fejl: {str(e)}"}