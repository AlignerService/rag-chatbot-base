
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3
import os
import openai

app = FastAPI()

class Payload(BaseModel):
    question: str
    ticketId: str

@app.get("/")
def read_root():
    return {"message": "Service is running"}

@app.post("/answer")
def generate_answer(payload: Payload):
    # Simulate correct return
    return {"reply": f"Spørgsmål modtaget: {payload.question}"}
