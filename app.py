import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from openai import OpenAI

app = FastAPI()

# Initial token setup
zoho_token = os.getenv("ZOHO_ACCESS_TOKEN")
refresh_token = os.getenv("ZOHO_REFRESH_TOKEN")
client_id = os.getenv("ZOHO_CLIENT_ID")
client_secret = os.getenv("ZOHO_CLIENT_SECRET")
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class TicketRequest(BaseModel):
    ticketId: str
    question: str

def refresh_access_token():
    global zoho_token
    token_url = "https://accounts.zoho.eu/oauth/v2/token"
    params = {
        "refresh_token": refresh_token,
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "refresh_token"
    }
    response = requests.post(token_url, params=params)
    if response.status_code == 200:
        zoho_token = response.json().get("access_token", "")
    else:
        raise Exception("Failed to refresh ZoHo access token")

@app.post("/answer")
async def generate_answer(request: TicketRequest):
    global zoho_token
    ticket_id = request.ticketId
    user_question = request.question

    # Attempt to fetch ZoHo ticket data
    zoho_url = f"https://desk.zoho.eu/api/v1/tickets/{ticket_id}/threads"
    headers = {"Authorization": f"Zoho-oauthtoken {zoho_token}"}
    response = requests.get(zoho_url, headers=headers)

    # If token is invalid, refresh it and retry once
    if response.status_code == 401:
        refresh_access_token()
        headers["Authorization"] = f"Zoho-oauthtoken {zoho_token}"
        response = requests.get(zoho_url, headers=headers)

    if response.status_code != 200:
        return JSONResponse(
            status_code=500,
            content={"reply": "Kunne ikke hente ticket-data fra ZoHo."}
        )

    threads = response.json().get("data", [])
    full_thread_text = "\n\n".join([t.get("content", "") for t in threads])

    # Generate AI answer
    prompt = f"Ticket historik:\n{full_thread_text}\n\nSpørgsmål: {user_question}\n\nSvar:"
    chat_completion = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Du er en hjælpsom AI-assistent for AlignerService."},
            {"role": "user", "content": prompt}
        ]
    )

    final_answer = chat_completion.choices[0].message.content.strip()
    return {"reply": final_answer}