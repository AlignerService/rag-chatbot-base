
import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from openai import OpenAI

app = FastAPI()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class TicketRequest(BaseModel):
    ticketId: str
    question: str

@app.post("/answer")
async def generate_answer(request: TicketRequest):
    ticket_id = request.ticketId
    user_question = request.question

    # 1. Forny token hvis nødvendigt
    refresh_token = os.getenv("ZOHO_REFRESH_TOKEN")
    client_id = os.getenv("ZOHO_CLIENT_ID")
    client_secret = os.getenv("ZOHO_CLIENT_SECRET")

    # Brug gemt token hvis den findes, ellers forny
    access_token = os.getenv("ZOHO_ACCESS_TOKEN")

    zoho_url = f"https://desk.zoho.eu/api/v1/tickets/{ticket_id}/threads"
    headers = {"Authorization": f"Zoho-oauthtoken {access_token}"}

    response = requests.get(zoho_url, headers=headers)

    # 👇 Print hele responsen fra ZoHo
    print("ZoHo API response:")
    print(f"Status code: {response.status_code}")
    print(f"Body: {response.text}")

    if response.status_code != 200:
        return JSONResponse(
            status_code=500,
            content={"reply": "Kunne ikke hente ticket-data fra ZoHo."}
        )

    threads = response.json().get("data", [])
    full_thread_text = "\n\n".join([t.get("content", "") for t in threads])

    # 2. Spørg OpenAI
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
