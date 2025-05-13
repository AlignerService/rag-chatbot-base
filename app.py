
import os
import requests
from fastapi import FastAPI, Request

app = FastAPI()

ZOHO_CLIENT_ID = os.getenv("ZOHO_CLIENT_ID")
ZOHO_CLIENT_SECRET = os.getenv("ZOHO_CLIENT_SECRET")
ZOHO_REFRESH_TOKEN = os.getenv("ZOHO_REFRESH_TOKEN")
ZOHO_ORGID = os.getenv("ZOHO_ORGID")

def get_access_token():
    url = "https://accounts.zoho.eu/oauth/v2/token"
    params = {
        "refresh_token": ZOHO_REFRESH_TOKEN,
        "client_id": ZOHO_CLIENT_ID,
        "client_secret": ZOHO_CLIENT_SECRET,
        "grant_type": "refresh_token"
    }
    response = requests.post(url, params=params)
    if response.status_code == 200:
        return response.json().get("access_token")
    else:
        print("ZoHo Refresh Token Response:")
        print("Status code:", response.status_code)
        print("Body:", response.text)
        return None

@app.post("/answer")
async def generate_answer(request: Request):
    data = await request.json()
    ticket_id = data.get("ticketId")
    question = data.get("question")

    access_token = get_access_token()
    if not access_token:
        return {"reply": "Kunne ikke opdatere ZoHo access token."}

    zoho_url = f"https://desk.zoho.eu/api/v1/tickets/{ticket_id}"
    headers = {
        "Authorization": f"Zoho-oauthtoken {access_token}",
        "orgId": ZOHO_ORGID
    }
    response = requests.get(zoho_url, headers=headers)
    if response.status_code == 200:
        ticket_data = response.json()
        subject = ticket_data.get("subject", "Intet emne")
        description = ticket_data.get("description", "Ingen beskrivelse")
        return {"reply": f"Sagens emne: {subject}\nBeskrivelse: {description}"}
    else:
        print("ZoHo API response:")
        print("Status code:", response.status_code)
        print("Body:", response.text)
        return {"reply": "Kunne ikke hente ticket-data fra ZoHo."}
