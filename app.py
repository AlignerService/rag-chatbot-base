
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "RAG system is live"}

@app.post("/answer")
async def answer(request: Request):
    data = await request.json()
    question = data.get("question", "")
    return {"answer": f"Du spurgte: {question}"}
