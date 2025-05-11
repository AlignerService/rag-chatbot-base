from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

@app.post("/chat")
async def chat_endpoint(request: Request):
    try:
        data = await request.json()
        print("✅ Data received:", data)
        reply = f"Tak for din besked: {data.get('prompt', 'Ingen prompt modtaget')}"
        return {"reply": reply}
    except Exception as e:
        print("❌ Error:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})
