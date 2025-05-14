
from fastapi import FastAPI, Request

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "OK", "message": "Service is running."}

@app.post("/answer")
async def answer(request: Request):
    data = await request.json()
    question = data.get("question", "")
    return {"answer": f"Du spurgte: '{question}', men dette er kun et test-svar."}
