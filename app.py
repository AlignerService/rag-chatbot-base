
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import openai
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai.api_key = os.getenv("OPENAI_API_KEY")

def get_ticket_thread(ticket_id: str, access_token: str) -> str:
    url = f"https://desk.zoho.eu/api/v1/tickets/{ticket_id}/conversations"
    headers = {
        "Authorization": f"Zoho-oauthtoken {access_token}"
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return "Failed to retrieve ticket thread."

    data = response.json()
    messages = [msg.get("content", "") for msg in data.get("data", [])]
    return "\n---\n".join(messages)

@app.post("/chat")
async def chat_endpoint(request: Request):
    body = await request.json()
    ticket_id = body.get("ticketId")
    question = body.get("question", "")

    access_token = os.getenv("ZOHO_ACCESS_TOKEN")
    if not access_token:
        return {"reply": "Zoho access token not configured."}

    ticket_thread = get_ticket_thread(ticket_id, access_token)

    client = openai.OpenAI()

    chat_completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant for a dental support service."},
            {
                "role": "user",
                "content": "Thread:\n" + ticket_thread + "\n\nQuestion:\n" + question
            }
        ]
    )

    return {"reply": chat_completion.choices[0].message.content}
