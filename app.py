from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS: Tillad alt i testmiljø
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/answer")
async def answer(request: Request):
    data = await request.json()
    question = data.get("question", "Intet spørgsmål modtaget.")
    return {"answer": f"Du spurgte: '{question}'. Dette er et testsvar fra serveren."}
