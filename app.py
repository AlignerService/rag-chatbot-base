from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/chat")
async def test_webhook():
    return {"message": "Webhook endpoint is working ✅"}

@app.post("/chat")
async def chat_webhook(request: Request):
    try:
        body = await request.body()
        print("📥 RAW REQUEST BODY:")
        print(body.decode("utf-8"))

        json_data = await request.json()
        print("✅ PARSED JSON:")
        print(json_data)

        ticket_id = json_data.get("ticketId", "N/A")
        prompt = json_data.get("prompt", "")
        print("🧠 Prompt:", prompt)

        # Simuleret svar
        response = {
            "reply": f"Tak for din besked om sag {ticket_id}. Vi vender tilbage snarest!"
        }

        print("📤 RESPONSE SENT:")
        print(response)
        return JSONResponse(content=response)

    except Exception as e:
        print("❌ Exception:", str(e))
        return JSONResponse(status_code=500, content={"error": str(e)})
