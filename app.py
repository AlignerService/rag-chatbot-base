import os
import requests
import logging
from fastapi import FastAPI, Request

app = FastAPI()

@app.post("/answer")
async def generate_answer(request: Request):
    data = await request.json()
    ticket_id = data.get("ticketId")
    question = data.get("question")

    # Get environment variables
    access_token = os.getenv("ZOHO_ACCESS_TOKEN")
    refresh_token = os.getenv("ZOHO_REFRESH_TOKEN")
    client_id = os.getenv("ZOHO_CLIENT_ID")
    client_secret = os.getenv("ZOHO_CLIENT_SECRET")
    redirect_uri = os.getenv("ZOHO_REDIRECT_URI")
    org_id = os.getenv("ZOHO_ORGID")

    # Validate required variables
    if not all([refresh_token, client_id, client_secret, redirect_uri]):
        return {"reply": "Manglende miljøvariabler."}

    # Attempt to refresh access token
    token_url = "https://accounts.zoho.eu/oauth/v2/token"
    params = {
        "refresh_token": refresh_token,
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "refresh_token",
        "redirect_uri": redirect_uri,
    }
    response = requests.post(token_url, params=params)
    
    logging.warning("ZoHo Refresh Token Response:")
    logging.warning(f"Status code: {response.status_code}")
    logging.warning(f"Body: {response.text}")

    if response.status_code != 200:
        return {"reply": "Kunne ikke opdatere ZoHo access token."}

    new_access_token = response.json().get("access_token")
    if not new_access_token:
        return {"reply": "Access token ikke fundet i ZoHo-svaret."}

    # Use the new access token to fetch ticket data
    headers = {
        "Authorization": f"Zoho-oauthtoken {new_access_token}",
        "orgId": org_id,
    }
    zoho_url = f"https://desk.zoho.eu/api/v1/tickets/{ticket_id}"
    ticket_response = requests.get(zoho_url, headers=headers)

    if ticket_response.status_code != 200:
        return {"reply": "Kunne ikke hente ticket-data fra ZoHo."}

    ticket_data = ticket_response.json()
    description = ticket_data.get("description", "Ingen beskrivelse.")

    return {"reply": f"Beskrivelse af sagen: {description}"}