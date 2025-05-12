import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from openai import OpenAI

app = FastAPI()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
zoho_token = os.getenv("ZOHO_TOKEN")

class TicketRequest(BaseModel):
    ticketId: str
    question: str

@app.post("/answer")
async def generate_answer(request: TicketRequest):
    ticket_id = request.ticketId
    user_question = request.question

    # 1. Hent hele ticket-tråden fra ZoHo
    zoho_url = f"https://desk.zoho.eu/api/v1/tickets/{ticket_id}/threads"
    headers = {"Authorization": f"Zoho-oauthtoken {zoho_token}"}
    response = requests.get(zoho_url, headers=headers)

    # → Fejl? Returnér debug-information
    if response.status_code != 200:
        return JSONResponse(
            status_code=500,
            content={
                "reply": "Kunne ikke hente ticket-data fra ZoHo.",
                "status_code": response.status_code,
                "zoho_response": response.text
            }
        )

    # 2. Saml ticket-historikken
    threads = response.json().get("data", [])
    if not threads:
        return JSONResponse(
            status_code=404,
            content={
                "reply": "Ingen tråde fundet for dette ticket ID.",
                "ticketId": ticket_id
            }
        )

    full_thread_text = "\n\n".join([t.get("content", "") for t in threads])

    # 3. Send prompt til OpenAI
    prompt = f"Ticket historik:\n{full_thread_text}\n\nSpørgsmål: {user_question}\n\nSvar:"
    chat_completion = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Du er en hjælpsom AI-assistent for AlignerService."},
            {"role": "user", "content": prompt}
        ]
    )

    final_answer = chat_completion.choices[0].message.content.strip()
    return {"reply": final_answer}"
