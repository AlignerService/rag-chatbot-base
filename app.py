from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

@app.post("/chat")
async def chat_proxy(request: Request):
    try:
        body = await request.body()
        print("📥 RAW REQUEST BODY:")
        print(body.decode("utf-8"))

        json_data = await request.json()
        print("✅ PARSED JSON:")
        print(json_data)

        ticket_id = json_data.get("ticketId")
        prompt = json_data.get("prompt")

        if not ticket_id or not prompt:
            return JSONResponse(status_code=400, content={"error": "Missing 'ticketId' or 'prompt'"})

        reply = f"(Dummy response): You asked: '{prompt}'"
        response_data = {
            "ticketId": ticket_id,
            "reply": reply
        }

        print("📤 RESPONSE SENT:")
        print(response_data)

        return response_data

    except Exception as e:
        print("❌ Exception:", str(e))
        return JSONResponse(status_code=500, content={"error": str(e)})
