from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

@app.get("/")
def root():
    return {"message": "RAG is running"}

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_input = data.get("question", "")
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": user_input}],
        max_tokens=500
    )
    return JSONResponse({"answer": response.choices[0].message["content"]})