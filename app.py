
import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from openai import OpenAI

app = FastAPI()

# Load environment variables
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
zoho_access_token = os.getenv("ZOHO_ACCESS_TOKEN")
zoho_refresh_token = os.getenv("ZOHO_REFRESH_TOKEN")
zoho_client_id = os.getenv("ZOHO_CLIENT_ID")
zoho_client_secret = os.getenv("ZOHO_CLIENT_SECRET")
zoho_org_id = os.getenv("ZOHO_ORGID")

class TicketRequest(BaseModel):
    ticketId: str
    question: str

def refresh_zoho_token():
    url = "https://accounts.zoho.eu/oauth/v2/token"
    params = {
        "refresh_token": zoho_refresh_token,
        "client_id": zoho_client_id,
        "client_secret": zoho_client_secret,
        "grant_type": "refresh_token"
    }
    response = requests.post(url, params=params)
    if response.status_code == 200:
        new_token = response.json().get("access_token")
        return new_token
    return None

def fetch_ticket_threads(ticket_id, token):
    url = f"https://desk.zoho.eu/api/v1/tickets/{ticket_id}/threads"
    headers = {
        "Authorization": f"Zoho-oauthtoken {token}",
        "orgId": zoho_org_id
    }
    response = requests.get(url, headers=headers)
    return response

@app.post("/answer")
async def generate_answer(request: TicketRequest):
    global zoho_access_token
    ticket_id = request.ticketId
    user_question = request.question

    # Attempt to fetch ticket threads
    response = fetch_ticket_threads(ticket_id, zoho_access_token)
    if response.status_code == 401:
        zoho_access_token = refresh_zoho_token()
        if not zoho_access_token:
            return JSONResponse(status_code=500, content={"reply": "Kunne ikke opdatere ZoHo access token."})
        response = fetch_ticket_threads(ticket_id, zoho_access_token)

    if response.status_code != 200:
        return JSONResponse(status_code=500, content={"reply": "Kunne ikke hente ticket-data fra ZoHo."})

    threads = response.json().get("data", [])
    full_thread_text = "\n\n".join([t.get("content", "") for t in threads])

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
