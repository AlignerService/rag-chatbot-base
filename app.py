from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

# 👇 Denne route gør ZoHo glad (GET til /chat virker nu)
@app.get("/chat")
async def chat_healthcheck():
    return {"message": "Webhook endpoint is working ✅"}

# 👇 Denne route håndterer ZoHo POST-requests som tidligere
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

        if not ticket_id:
            return JSONResponse(status_code=400, content={"error": "ticketId mangler i request"})

        print("🎯 Klar til behandling af ticket:", ticket_id)

        return {"reply": "Alt ser godt ud ✅", "ticketId": ticket_id}

    except Exception as e:
        print("❌ Exception:", str(e))
        return JSONResponse(status_code=500, content={"error": str(e)})
