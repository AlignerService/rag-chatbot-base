
import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from openai import OpenAI

app = FastAPI()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load ZoHo environment variables
zoho_access_token = os.getenv("ZOHO_ACCESS_TOKEN")
zoho_refresh_token = os.getenv("ZOHO_REFRESH_TOKEN")
zoho_client_id = os.getenv("ZOHO_CLIENT_ID")
zoho_client_secret = os.getenv("ZOHO_CLIENT_SECRET")
zoho_redirect_uri = os.getenv("ZOHO_REDIRECT_URI")

class TicketRequest(BaseModel):
    ticketId: str
    question: str

def refresh_access_token():
    token_url = "https://accounts.zoho.eu/oauth/v2/token"
    params = {
        "refresh_token": zoho_refresh_token,
        "client_id": zoho_client_id,
        "client_secret": zoho_client_secret,
        "redirect_uri": zoho_redirect_uri,
        "grant_type": "refresh_token"
    }
    response = requests.post(token_url, params=params)
    if response.status_code == 200:
        new_token = response.json().get("access_token")
        if new_token:
            global zoho_access_token
            zoho_access_token = new_token
            return True
    return False

@app.post("/answer")
async def generate_answer(request: TicketRequest):
    ticket_id = request.ticketId
    user_question = request.question

    # 1. Hent hele ticket-tråden fra ZoHo
    zoho_url = f"https://desk.zoho.eu/api/v1/tickets/{ticket_id}/threads"
    headers = {"Authorization": f"Zoho-oauthtoken {zoho_access_token}"}
    response = requests.get(zoho_url, headers=headers)

    # Hvis token er udløbet, prøv at forny det én gang
    if response.status_code == 401:
        if refresh_access_token():
            headers = {"Authorization": f"Zoho-oauthtoken {zoho_access_token}"}
            response = requests.get(zoho_url, headers=headers)
        else:
            return JSONResponse(status_code=500, content={"reply": "Kunne ikke opdatere ZoHo access token."})

    if response.status_code != 200:
        return JSONResponse(status_code=500, content={"reply": "Kunne ikke hente ticket-data fra ZoHo."})

    threads = response.json().get("data", [])
    full_thread_text = "\n\n".join([t.get("content", "") for t in threads])

    # 2. Spørg OpenAI med historik og spørgsmål
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
