from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import openai
import os

app = FastAPI()

# Hent API-nøglen fra miljøvariabler
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.post("/chat")
async def chat_proxy(request: Request):
    try:
        # Læs og vis raw body
        body = await request.body()
        print("📥 RAW REQUEST BODY:")
        print(body.decode("utf-8"))

        # Parse JSON
        json_data = await request.json()
        print("✅ PARSED JSON:")
        print(json_data)

        ticket_id = json_data.get("ticketId")
        if not ticket_id:
            return JSONResponse(status_code=400, content={"error": "ticketId mangler i request"})

        # Kald OpenAI API
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant for dentists working with clear aligner treatments."
            },
            {
                "role": "user",
                "content": f"Support request from ticket ID {ticket_id}. Please provide a helpful reply."
            }
        ]

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=300
        )

        reply = response.choices[0].message.content.strip()
        response_payload = {
            "message": reply,
            "ticketId": ticket_id
        }

        print("📤 RESPONSE SENT:")
        print(response_payload)

        return response_payload

    except Exception as e:
        print("❌ Exception:", str(e))
        return JSONResponse(status_code=500, content={"error": str(e)})
