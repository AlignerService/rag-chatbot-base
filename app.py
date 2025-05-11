
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
        if not ticket_id:
            return JSONResponse(status_code=400, content={"error": "ticketId mangler i request"})

        return {"message": "Alt ser godt ud", "ticketId": ticket_id}

    except Exception as e:
        print("❌ Exception:", str(e))
        return JSONResponse(status_code=500, content={"error": str(e)})
