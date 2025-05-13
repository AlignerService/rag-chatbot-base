import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from openai import OpenAI

app = FastAPI()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
zoho_token = os.getenv("ZOHO_ACCESS_TOKEN")
zoho_org_id = os.getenv("ZOHO_ORGID")

class TicketRequest(BaseModel):
    ticketId: str
    question: str

@app.post("/answer")
async def generate_answer(request: TicketRequest):
    ticket_id = request.ticketId
    user_question = request.question

    # 1. Hent hele ticket-tråden fra ZoHo
    zoho_url = f"https://desk.zoho.eu/api/v1/tickets/{ticket_id}/threads"
    headers = {
        "Authorization": f"Zoho-oauthtoken {zoho_token}",
        "orgId": zoho_org_id
    }
    response = requests.get(zoho_url, headers=headers)

    if response.status_code != 200:
        return JSONResponse(
            status_code=500,
            content={"reply": "Kunne ikke hente ticket-data fra ZoHo."}
        )

    threads = response.json().get("data", [])
    full_thread_text = "

".join([t.get("content", "") for t in threads])

    # 2. Spørg OpenAI med historik og spørgsmål
    prompt = f"Ticket historik:
{full_thread_text}

Spørgsmål: {user_question}

Svar:"
    chat_completion = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Du er en hjælpsom AI-assistent for AlignerService."},
            {"role": "user", "content": prompt}
        ]
    )

    final_answer = chat_completion.choices[0].message.content.strip()
    return {"reply": final_answer}
