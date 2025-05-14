
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS (tillader alle domæner, kun til testformål)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "RAG API is running."}

@app.post("/answer")
async def answer(request: Request):
    data = await request.json()
    question = data.get("question", "intet spørgsmål angivet")
    return {"answer": f"Du spurgte: {question}"}
